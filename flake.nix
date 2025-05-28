{
  description = "nanoGPT training packaged as a Nix flake";

  inputs = {
    # the exact state of the used packages in here depends
    # on the commit from this branch in the flake.lock file
    # you can inspect this with the command `nix flake metadata`
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  # run `nix flake show` to see all the stuff that's defined in here
  # and provided as some kind of output
  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
    lib = pkgs.lib;
    selectedPython = pkgs.python3;
    # for ROCM use torchWithRocm and add --compile=False flag to the train command
    selectedTorch = selectedPython.pkgs.torch;

    # define a seperate python environemnt for each step
    # so we do not depend on extra stuff we don't actually use use in a given step
    pythonEnv = selectedPython.withPackages (ps: with ps; [
      selectedTorch
      numpy
      transformers
      datasets
      tiktoken
      wandb
      tqdm
    ]);
    preprocessingPythonEnv = selectedPython.withPackages (ps: with ps; [
      numpy
      tiktoken
    ]);
    inferencePythonEnv = selectedPython.withPackages (ps: with ps; [
      selectedTorch
      numpy
      tiktoken
    ]);
    # fetch data in a sandobx of its own which provided
    # 1. a known identify of that dataset in terms of commit hash
    training-data-git-rev = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e";
    training-data = pkgs.fetchurl {
      url = "https://raw.githubusercontent.com/karpathy/char-rnn/${training-data-git-rev}/data/tinyshakespeare/input.txt";
      # a deterministic expected outcome of the fetching process
      # (this is why network acess is allowed at all here)
      hash = "sha256-hsTmqp23wELsefM53LltQrAHXha4/C6Gvwylfi3FZe0=";
    };
    pre-processing = pkgs.stdenv.mkDerivation {
        name = "shakespeare-char-preprocessing";
        src = ./data/shakespeare_char;
        buildInputs = [
          preprocessingPythonEnv
        ];
        postPatch = ''
            # we need to replace those paths relative to source
            # because store paths in which our code runs are not writable
            substituteInPlace prepare.py \
              --replace-fail "os.path.dirname(__file__)" "'$PWD'"
        '';
        buildPhase = ''
            # add input data in expected place
            ln -s ${training-data} input.txt
            # prepare output directory
            mkdir -p $out
            # actually prepare training data
            python prepare.py
        '';
        installPhase = ''
            # copy preparation output to build step output
            cp train.bin $out/
            cp val.bin $out/
            cp meta.pkl $out/
        '';
      };
    outlier-removal = pkgs.runCommand
    "shakespeare-outlier-removal" {
        buildInputs = [
          (pkgs.python3.withPackages (ps: with ps; [
            numpy
          ]))
        ];
      } ''
        # copy preprocessed data from previous step
        cp ${pre-processing}/train.bin .
        cp ${pre-processing}/val.bin .
        cp ${pre-processing}/meta.pkl .

        # remove outliers from the tokenized data
        python ${./additional_preprocessing/remove_outliers.py}

        mkdir -p $out
        cp train_filtered.bin $out/train.bin
        cp val_filtered.bin $out/val.bin
        cp meta_filtered.pkl $out/meta.pkl
      '';

    training = pkgs.stdenv.mkDerivation {
        name = "shakespeare-char-training";
        src = ./.;
        buildInputs = [ pythonEnv ];
        # TODO: make determinstic
        # see https://pytorch.org/docs/stable/notes/randomness.html
        # TODO: add GPU support
        # see https://github.com/NixOS/nixpkgs/pull/256230 and related efforts
        # TODO: add meta attribute to describe types of ML operations
        # see: https://nixos.org/manual/nixpkgs/stable/#chap-meta
        buildPhase = ''
            # copy the filtered training data
            mkdir -p data/shakespeare_char
            cp ${outlier-removal}/train.bin data/shakespeare_char/
            cp ${outlier-removal}/val.bin data/shakespeare_char/
            cp ${outlier-removal}/meta.pkl data/shakespeare_char/

            # set output directory
            export OUT_DIR=$PWD/out-shakespeare-char
            mkdir -p $OUT_DIR
  
            # run training with deterministic settings
            python train.py config/train_shakespeare_char.py \
              --device=cpu \
              --compile=False \
              --eval_iters=20 \
              --log_interval=1 \
              --block_size=64 \
              --batch_size=12 \
              --n_layer=4 \
              --n_head=4 \
              --n_embd=128 \
              --max_iters=2000 \
              --lr_decay_iters=2000 \
              --dropout=0.0
        '';
        installPhase = ''
            mkdir -p $out
            # copy the trained model and logs
            cp -r out-shakespeare-char/* $out/
            # also copy the meta.pkl file needed for inference
            cp data/shakespeare_char/meta.pkl $out/
        '';
      };
  in
   {
    packages.x86_64-linux.default = training;

    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [ inferencePythonEnv ];
      shellHook = ''
        echo "python environment for inference ready"
        echo "use `nix build` command to symlink training result into result/ folder"
      '';
    };
  };
}

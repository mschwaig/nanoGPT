{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

# docs : https://pytorch.org/docs/stable/notes/randomness.html

  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
    lib = pkgs.lib;
    pythonEnv = pkgs.python3.withPackages (ps: with ps; [
      # for ROCM use torchWithRocm and add --compile=False flag to the train command
      torchWithRocm
      # for nvidia use regular torch
      # torch
      numpy
      transformers
      datasets
      tiktoken
      wandb
      tqdm
    ]);
    preprocessingPythonEnv = pkgs.python3.withPackages (ps: with ps; [
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
          # define a seperate python environemnt for this
          # so we do not depend on extra stuff we don't use in this step
          (pkgs.python3.withPackages (ps: with ps; [
            numpy
            tiktoken
          ]))
        ];
        postPatch = ''
            # we need to replace those paths relative to source
            # because store paths in which our code runs are not writable
            substituteInPlace prepare.py \
              --replace "os.path.dirname(__file__)" "'$PWD'"
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
    outlier-removal = pkgs.stdenv.mkDerivation {
        name = "shakespeare-outlier-removal";
        src = ./data/shakespeare_char;
        buildInputs = [
          (pkgs.python3.withPackages (ps: with ps; [
            numpy
          ]))
        ];
        postPatch = ''
            # fix paths to work in build directory
            substituteInPlace remove_outliers.py \
              --replace "os.path.dirname(__file__)" "'$PWD'"
        '';
        buildPhase = ''
            # copy preprocessed data from previous step
            cp ${pre-processing}/train.bin .
            cp ${pre-processing}/val.bin .
            cp ${pre-processing}/meta.pkl .
            
            # remove outliers from the tokenized data
            python remove_outliers.py
        '';
        installPhase = ''
            mkdir -p $out
            cp train_filtered.bin $out/train.bin
            cp val_filtered.bin $out/val.bin
            cp meta_filtered.pkl $out/meta.pkl
        '';
      };
  in
   {

    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [ pythonEnv outlier-removal ];
    };
  };
}

{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

# docs : https://pytorch.org/docs/stable/notes/randomness.html

  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
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
  in
   {

    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [ pythonEnv ];
    };
  };
}

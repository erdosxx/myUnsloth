{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-unstable-small";
    };
    # nixpkgs.url = "nixpkgs";
    /* nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; */
    # nixpkgs.url = "github:NixOS/nixpkgs?&rev=632d9a851e1db6736135f73df4dae76469a72168";
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    poetry2nix = {
      # url = "github:nix-community/poetry2nix?&rev=b90fbfbb71d4da8de2d8e4dba0fa85c7cf07015f";
      url = "github:nix-community/poetry2nix";
      # url =
      #   "github:nix-community/poetry2nix?&rev=8c25e871bba3f472e1569bbf6c0f52dcc34bf2a4";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };

          /* overlays = [ xformers ]; */
        };

        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; })
          mkPoetryApplication defaultPoetryOverrides;

        # Ex: pypkgs-build-requirements = { textgrad = [ "setuptools" ]; };
        pypkgs-build-requirements = {
          xformers = [ "setuptools" ];
          unsloth = [ "setuptools" ];
        };

        p2n-overrides = defaultPoetryOverrides.extend (final: prev:
          builtins.mapAttrs
            (package: build-requirements:
              (builtins.getAttr package prev).overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or [ ]) ++ (builtins.map
                  (pkg:
                    if builtins.isString pkg then
                      builtins.getAttr pkg prev
                    else
                      pkg)
                  build-requirements);
              }))
            pypkgs-build-requirements);
        python_ver = "python312";
        pythonPackages = pkgs.${python_ver + "Packages"};
      in
      {
        packages = {
          myapp = mkPoetryApplication {
            projectDir = self;
            # projectDir = ./.;
            python = pkgs.${python_ver};
            preferWheels = true;
            overrides = p2n-overrides;
          };
          default = self.packages.${system}.myapp;
        };


        # Shell for app dependencies.
        #
        #     nix develop
        #
        # Use this shell for developing your app.
        devShells.default =
          pkgs.mkShell { inputsFrom = [ self.packages.${system}.myapp ]; };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          inputsFrom = [
            self.packages.${system}.myapp
          ];

          packages = with pkgs; [
            # (poetry.override { python3 = python311; })
            poetry
            # for qtconsole
            pythonPackages.pyside2
            /* pythonPackages.xformers */
            /* pythonPackages.torch */
            /* pythonPackages.torchvision */
            /* pythonPackages.torch */
            /* cudatoolkit */
            linuxPackages.nvidia_x11
            qt5Full
            gnugrep
            gawk
            pass
          ];
          shellHook = ''
            alias vi="nvim"
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
            just qt
          '';
        };
      });
}

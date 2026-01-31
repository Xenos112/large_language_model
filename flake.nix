{
  description = "my-own-llm dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # Python 3.14 should be available in unstable by now
        python = pkgs.python314;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.uv
            pkgs.pkg-config
            pkgs.openssl
            pkgs.gcc
            pkgs.gnumake
            # For torch GPU support (optional - see note below)
            pkgs.linuxPackages.nvidia_x11
          ];

          shellHook = ''
            echo "🐍 Python: ${python.version}"
            echo "📦 uv: $(uv --version)"
            
            # Tell uv which Python to use
            export UV_PYTHON="${python}/bin/python3"
            
            # Set up for potential CUDA usage with torch
            export CUDA_PATH="${pkgs.linuxPackages.nvidia_x11}"
            
            # Auto-create venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            
            echo ""
            echo "Run: uv pip install -e .   # to install deps"
            echo "Then: uv run python your_script.py"
          '';
        };
      });
}

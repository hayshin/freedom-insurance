{
  pkgs,
  lib,
  config,
  ...
}:
{
  # https://devenv.sh/packages/
  packages = [
    pkgs.jupyter
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    
  ];
  

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };

  # https://devenv.sh/processes/
  processes.jupyter.exec = "jupyter notebook";

  # See full reference at https://devenv.sh/reference/options/
}

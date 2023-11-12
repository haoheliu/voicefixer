# Changelog

## 2023-11-12
### by @[francocm](https://github.com/francocm)

1. Integrated fixes by @[manmay-nakhashi](https://github.com/manmay-nakhashi) which include support for newer `librosa`. This is from PR https://github.com/haoheliu/voicefixer/pull/56 which was not merged to `main` yet.
2. Fixed compatibility with newer NumPy interfaces, as `nn.utils.weight_norm` is now `nn.utils.parametrizations.weight_norm`.
3. Added ability to initialise weights only - useful to cache the setup without processing any files, via a new argument switch `--weight_prepare`.
4. Added `Dockerfile` that allows a clear easy replicable way of running the tool with exact environment setup.
5. Docs fix: Clarified what the run modes mean.
6. Docs fix: Removed hardcoded references to version 0.1.2 as these are not relevant from a docs perspective and by default latest should be used which is the default behaviour when version is not specified
7. Docs fix: Moved Changelog into a separate `CHANGELOG.md` file.

## 2023-10-20
### by @[manmay-nakhashi](https://github.com/manmay-nakhashi)

- 2023-10-20: Fix to support newer `librosa`

## 2023-09-03
### by @[haoheliu](https://github.com/haoheliu)

- Fix bugs on commandline voicefixer for windows users.

## 2023-08-18
### by @[haoheliu](https://github.com/haoheliu)

- Add commandline voicefixer tool to the pip package.
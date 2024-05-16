# 画像をffmpeg形式の動画にするプログラム

## environment
python 3.10.11 \
poetry 1.8.2

## Installation

1. Instaling Poetry

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    or

    ```bash
    wget -qO - https://install.python-poetry.org | python3 -
    ```

    Make sure that the PATH is passed to "$HOME/.local/bin".

    Or, make sure the following settings are in ".bashrc":

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

2. Check poetry install

    ```bash
    poetry --version
    ```

### 1. Preparation
```bash
pyenv install 3.10.11
```
```bash
cd ffmpeg-visualize
```
```bash
pyenv local 3.10.11
```

```bash
poetry install --no-root
```

### 2. visualize

```bash
poetry run python visualize.py \
-i ${HOME}/ffmpeg-visualize/img \
-o ${HOME}/ffmpeg-visualize/results 
```


results are in `results/detect.mp4`


### Optional

* [OPTIONAL] Genarate Panorama (projected) images and tracking

    Example:

    ```bash
    poetry run python visualize.py \
    -i ${HOME}/ffmpeg-visualize/img \
    -o ${HOME}/ffmpeg-visualize/results \
    -c ${HOME}/ffmpeg-visualize/track.txt
    ```

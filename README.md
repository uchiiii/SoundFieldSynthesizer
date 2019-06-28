# SoundFieldSynthesizer

### 全体の注意点
- ステレオ、モノラルチャネルなどを考えて接続すること.

### MADIface Pro 注意点 for windows
- 起動するときに接続した状態にしておくこと.
- ドライバーをインストールしてもコントロールパネルから認識されてないときは、MADIface Series SettingのWDM Devicesからデバイスを設定する. (MADIを使うときは、MADI(1+2)などのものを用いる.)
- pyaudioの通常版はASIOに対応していないので、(https://www.lfd.uci.edu/~gohlke/pythonlibs/)　ここからpyaudioをダウンロードすること.
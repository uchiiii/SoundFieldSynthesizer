# SoundFieldSynthesizer
前期実験におけるシミュレーション実験のコード

### コードの説明

#### src/cis.py
wave fileとnumpy間の読み出しと書き込みを行う.

#### src/dev.py
使用可能なサウンドデバイスを列挙する.

#### src/synthesizer3D.py
ある特定の周波数の駆動信号を求める. (メンバ関数：exploit_d)

#### src/plot-env.py
シミュレーションの条件を描画するためのもの.

#### src/simulator.py
src/synthesizer3D.pyを用いて、特定の周波数における理想的な周波数と今回の手法で生成された波形を画像として生成する.
![シミュレーション画像1](https://github.com/uchiiii/SoundFieldSynthesizer/examples/intersimu8.png)
![シミュレーション画像2](https://github.com/uchiiii/SoundFieldSynthesizer/exa    mples/intersimu9.png)

#### src/Tsynthesizer3D.py
全ての周波数に関して、周波数領域で駆動信号を求め、それらを逆フーリエ変換を用いて時間領域にし、時間領域でそれぞれのマイクの出力を求める.
![シミュレーション画像3](https://github.com/uchiiii/SoundFieldSynthesizer/exa    mples/intersimu10.png)

#### src/GUI.py
GUIを用いてリアルタイムでinteractiveに仮想的な音源の位置、つまり、駆動信号をかえれるようにしたかった. (これは未完成です)

#### src/RealtimeSynthesizer.py
標準入力から適宜仮想音源の位置を指定することで、リアルタイムで駆動信号を変更しながら出力する. waveファイルには書き込まない. 計算量がそれなりに多いので、現段階では途切れ途切れになってしまうことを確認した.(用いたPC : 1.6 GHz Intel Core i5)

### 全体の注意点
- ステレオ、モノラルチャネルなどを考えて接続すること.

### MADIface Pro 注意点 for MacOS
- 特にないのだが、うまくいかない時は、説明書にあるようにやり直すとうまくいく.

### MADIface Pro 注意点 for windows
- 起動するときに接続した状態にしておくこと.
- ドライバーをインストールしてもコントロールパネルから認識されてないときは、MADIface Series SettingのWDM Devicesからデバイスを設定する. (MADIを使うときは、MADI(1+2)などのものを用いる.)
- pyaudioの通常版はASIOに対応していないので、(https://www.lfd.uci.edu/~gohlke/pythonlibs/)　ここからpyaudioをダウンロードすること.

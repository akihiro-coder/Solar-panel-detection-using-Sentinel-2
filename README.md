# Solar-panel-detection-using-Sentinel-2
solafuneコンペ「低解像度衛星画像から太陽光パネルを検出する」
# 手法調査
- [機械学習法を用いたSPOT5/HRGデータの
土地被覆分類とその精度比較](https://www.ism.ac.jp/editsec/toukei/pdf/64-1-093.pdf)
- [機械学習による複数時期でのUAV河川空撮画像の地被分類手法の検討](https://www.jstage.jst.go.jp/article/jscejhe/75/2/75_I_667/_pdf)
- [河川の UAV 画像を対象とした機械学習による土地被覆分類手法の開発](http://www.constr.shibaura-it.ac.jp/constr/papers/me18122.pdf)
- [衛星画像を用いた深層学習による土地被覆土地利用分類](https://www.jstage.jst.go.jp/article/jiiars/1/1/1_135/_pdf/-char/ja)
- [Pi-SAR X2（航空機 SAR）× 深層学習による土地被覆分類](https://www.nict.go.jp/publication/shuppan/kihou-journal/houkoku65-1_HTML/2019R-03-03(13).pdf)
## まとめ
- random forest, svm, adaboosting, cnnが使われたみたい。
- 従来の機械学習手法では、画像特徴量が多いデータセットでは、random forest, svmの精度がboostingに比べて高く、逆に画像特徴量が少ないデータセットでは、boostingのほうが精度が高くなったらしい。
## その他の方法
- 深層学習と従来の機械学習を試そうと考えている
- 実際のセグメンテーションコンペ×衛星画像データセットで使われていたモデルについて調べてみると、Unet, highresolutionNetが使われていた。
  - [The 3rd Tellus Satellite Challenge実施！～入賞者たちのモデルに注目～](https://sorabatake.jp/10563/)
- また、当コンペで[Started Guideを示したディスカッションページ](https://solafune.com/ja/competitions/5dfc315c-1b24-4573-804f-7de8d707cd90?menu=discussion&tab=&page=1&topicId=54728653-0e25-4975-baad-6fe2f5185844)がある。
- このページでは、classic approachとしてrandom forest, svm, 深層学習としてU-Net, SegNet, DeepLabv3が紹介されている。
- また、個人的な興味にtransformerがある。transformerを応用してセマンティックセグメンテーションタスクが解けないかちょっと調べてみると、Segmenterというモデルを見つけた。[Segmenter: Transformer for Semantic Segmentation](https://paperswithcode.com/paper/segmenter-transformer-for-semantic)
### モデリングの選択
- 今後試すモデリング手法をいかに示す。
  1. random forest
  2. svm
  3. Unet
  4. SegNet
  5. DeepLabv3
  6. Segmenter

# random forest
- 決定木の最大の問題点は、訓練データに対して過剰適合してしまうこと
- ランダムフォレストはこの問題に対応する方法の1つ
- ランダムフォレストとは、少しずつ異なる決定木をたくさん集めたもの
- ランダムフォレストは、個々の決定木は比較的うまく予測できているが、一部のデータに対して過剰適合してしまっているという考えに基づいている
- それぞれ異なった方向に過剰適合した決定木を沢山作れば、その結果の平均を取ることで過剰適合の度合いを減らすことができる
- 決定木の予測性能を維持したまま、過剰適合が解決できることは厳密な数学で示すことが出来る。
- 上記の戦略を実装するためには、沢山の決定木を作らないといけないし、それぞれがある程度ターゲット値を予測できていて、さらにお互いに違っていなければならない
- ランダムフォレストという名前は、個々の決定木が互いに異なるように、決定木の構築過程で乱数を導入していることから付いている。
- ランダムフォレストに乱数を導入する方法は2つある。
    1. 決定木を作るためのデータポイントを選択する方法
    2. 分岐テストに用いる特徴を選択する方法
## random forestの構築

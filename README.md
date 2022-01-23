# agri-ner-bert
## 概要
文章から作物と思われるワードを検出するモデルです。
インタフェースは入力が文章に対し、出力が抽出したワードになることを目指します。
## ファイル構成
- src 
 - データセット・学習・推論器など
- test
 - 各モジュールのテスト
- config
 - nerモデルの設定
- data
 - 学習データやモデルを出力す
## データセットの形式
一列目が文章、二列目がスペース区切りの品目名です。
前処理中に正式なモデルの入力へと整形しています。
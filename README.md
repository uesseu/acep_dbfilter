# これはなに？
九州大学精神科発、ACEPのデータベースが吐き出したCSVを繋げるやつ。
このソフトはACEPの中心メンバーにしか役に立ちません。
依存先はpython3のみ。

オフラインで動かすのが前提なので、zipのみでイケるようにしています。
それ故、実装が少し汚いです。色々な妥協の産物です。
Ubuntu前提でしたが、Windowsでも動かせないかと思案中。
Macでも多分動きます。

# インストール
pipを使うのがいいです。

Windowsなら、まずはターミナルを開きます。
僕がサポートしている場合はwslは入っていると思いますが、
上手く行かなかったら九大ACEPコアメンバーの人は言って下さい。
適当な所にこのソフトをzipでダウンロードして、下記。

```
wsl
```

これでlinuxの環境に入ります。この状態で、

```
unzip ~/acep_dbfilter aced_dbfilter.zip
pip install -e acep_dbfilter
```

という感じで上手く行くかと。
上手く行かなければ、普通にzipをGUIで解凍して良いと思います。
上手く行かなければ教えて下さい。

# アップデート
残念ながらオフライン前提なので、アップデートは出来ません。
先ずは、このソフトをアンインストールして下さい。

```
pip install -e dbfilter
```

その上で、上記のインストールの手順を踏んで下さい。


ちなみに、オンラインの環境なら下記のようにアップデートできます。

```
wsl
cd ~/acep_dbfilter
git pull
```
テスト環境を作るとか、このパッケージをイジる場合はこれで。

# エンコーディング対策
MacとかLinuxとかWSLを使う場合はiconvコマンドを使うのが良いです。
だいたい、-fや-tで指定します。
'from-to'って事で分かりやすいですね。

```
cat iconv -f cp932 -t utf8
```

パイプで繋ぐことが出来るので、実質は下記のように使うかなと思う。

```
cat iconv -f cp932 | mergedrug drug.csv > result_withdrug.csv
```

ここで、分からないのがWindowsでどうかということ。
Git bashを使うとiconvも入るかも。今度調べます。

# 具体的な使い方
まずは、エクセルファイルをACEPのデータから取り出します。
その上で、それをCSVファイルに変換して下さい。
CSVファイルのエンコードにUTFを指定しないで下さい。
通常、cp932になるはずです。

必要物品
- 実験ファイル
- 薬剤ファイル
- 心理検査ファイルです。

それぞれ
- exp.csv
- drugs.csv
- psy.csv

という名前だとしてコードを書きます。

```sh
# まず、コードをUTFに変える。
cat exp.csv | iconv -f cp932 > exp-utf.csv
cat drugs.csv | iconv -f cp932 > drugs-utf.csv
cat psy.csv | iconv -f cp932 > psy-utf.csv

# 薬剤情報は巨大なので無駄を削ぐ
cat drugs-utf.csv | shrinkdrug > shrinked-drugs-utf.csv

# 薬剤情報と心理検査をくっつける
cat exp-utf.csv\
    | mergedrug shrinked-drugs-utf.csv\
    | mergepsycho panss-utf.csv > result.csv
```

# TODO
- 他の大学のフォーマットに対応する。
- Windows単体で動かしたい。

# FAQ
### ACEPのメンバーじゃないんだが
ACEPの中心メンバー以外の人が何か言っても返事しません。

### うん、分からん
ACEPの中心メンバーならメールしてください。

### テストコードはないのか？
僕の手元にある。

# これはなに？
九州大学精神科発、ACEPのデータベースが吐き出したCSVを繋げるやつ。
このソフトはACEPの中心メンバーにしか役に立ちません。
依存先はpython3のみ。

オフラインで動かすのが前提なので、zipのみでイケるようにしています。
それ故、実装が少し汚いです。
Ubuntu前提でしたが、Windowsでも動かせないかと思案中。

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

という感じで上手く行くかと。上手く行かなければ教えて下さい。

# アップデート
下記のようになります。

```
wsl
cd ~/acep_dbfilter
git pull
```

# エンコーディング対策
LinuxとかWSLを使う場合はiconvコマンドを使うのが良いです。
だいたい、-fや-tで指定します。
'from-to'って事で分かりやすいですね。

```
cat iconv -f cp932 -t utf8
```

パイプで繋ぐことが出来るので、実質は下記のように使うかなと思う。

```
cat iconv -f cp932 | mergedrug drug.csv > result_withdrug.csv
```

# 具体的な使い方
今度書く

# TODO
- READMEを詳しく書く


# FAQ
### ACEPのメンバーじゃないんだが
- ACEPの中心メンバー以外の人が何か言っても返事しません。

### うん、分からん
- ACEPの中心メンバーならメールしてください。

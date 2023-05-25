* * *

- [セットアップ](#セットアップ)
  - [ローカル開発環境](#ローカル開発環境)
    - [Pythonのバージョン](#pythonのバージョン)
    - [VSCode](#vscode)
      - [Extension](#extension)
      - [スニペットを使ったヘッダーの挿入](#スニペットを使ったヘッダーの挿入)
    - [リンタ・フォーマッタ](#リンタフォーマッタ)
    - [Pylanceのセットアップ](#pylanceのセットアップ)
    - [autoDocstring](#autodocstring)
- [コーディングルール](#コーディングルール)
  - [Docstring](#docstring)
  - [TypeHints](#typehints)
    - [TypeHintsの注意点](#typehintsの注意点)


# セットアップ

## ローカル開発環境

### Pythonのバージョン

Python >= 3.9.1

### VSCode

こだわりがなければ[VSCode](https://code.visualstudio.com/download)を推奨します。

#### Extension
以下のExtensionの追加を推奨します。

*   Trailing Spaces
*   Path Autocomplete
*   Python
*   Pylance
*   autoDocstring
*   Markdown All in One

#### スニペットを使ったヘッダーの挿入

[VSCocdeのスニペット機能](https://code.visualstudio.com/docs/editor/userdefinedsnippets#_builtin-snippets)を使用してヘッダーが挿入されるよう設定する。

1. VSCodeを開く。
1. 左上のCode→基本設定→ユーザースニペットの構成に移動し、pythonと入力する。
1. 以下の`@author`, `@contact`を自分のものに書き換え、python.jsonに貼り付ける。
    ```json
    {
        "HEADER": {
            "prefix": "header",
            "body": [
                "##",
                "# @file    :   $TM_FILENAME",
                "#",
                "# @brief   :   None",
                "#",
                "# @author  :   Tanaka Taro",
                "# @contact :   t.tanaka@walc.co.jp",
                "# @date    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE",
                "#",
                "# (C)Copyright $CURRENT_YEAR, WALC Inc.",
                "$0",
            ],
        }
    }
    ```

### リンタ・フォーマッタ

可読性の高いコードを記述する際に、PEP8に準拠したコードを書くことが要求されます。
自動でコードを整形してくれるFormatter(black)と、文法のチェックを行ってくれるLinter(flake8)をセットアップしましょう。

* flake8のインストール
    ```bassh
    pip install flake8
    ```

* blackのインストール
    ```bash
    pip install black
    ```

    パスを獲得する
    ```bash
    which black
    ```

* isort
    VSCodeデフォルトで組み込まれているので、インストールする必要なし

* VSCodeへのセットアップ

    * command + , でsettingを開く

    * flake8
        Editor: Format On Save”
        チェックを入れる
        “Python › Linting: Pylint Enabled”
        チェックを外す
        ”Python › Linting: Flake8 Enabled”
        チェックを入れる
        Python › Linting: Lint On Save
        チェックを入れる
        black(https://pyteyon.hatenablog.com/entry/2020/10/04/052716)
        ”Python › Formatting: Black Path”
        先ほど調べた black のフルパス
        ”Python › Formatting: Provider”
        * 文字列の長さ制限を設定する
        "Python › Formatting: Black Args"と検索し、`--max-line-length=119`を追加

    * black を選択
        ”Python › Formatting: Black Args”(https://dev.to/adamlombard/vscode-setting-line-lengths-in-the-black-python-code-formatter-1g62)
        Add itemをクリックし`--line-length=119`と追加


    * ”Editor: Format On Save”
        チェックを入れる

    * isort
        * 検索窓で”Editor: Code Actions On Save”
            ```json
            "editor.codeActionsOnSave": null
            ```

            ```json
            "editor.codeActionsOnSave": {
                    "source.organizeImports": true
                }
            ```

    * blackとisortのコンフリクトを解消するために、設定を行う。
        blackとisortはimportするライブラリの順番について異なるルールを適用するため、
        black適用 -> isort適当 -> black適当と無限に修正しつづける。
        今回はblackのルールを優先的に適用する。

        ```json
        "python.sortImports.args": [
            "--profile", "black"
        ],
        ```

### Pylanceのセットアップ

Pylanceは型アノテーションの静的解析や、IntelliCode(補完サポート)を行います。

1. VSCodeのExtensionでPylanceをインストールする
1. command + , でsettingを開く
1. Python › Analysis: Type Checking Modeで検索して、`basic`に設定する

### autoDocstring

1. command + , でsettingを開く
2. autoDocstringで検索しAutoDocstring: Docstring Formatを`google`に設定する


# コーディングルール

基本的には[PEP8](https://peps.python.org/pep-0008)に準拠します。VSCodeにFormatterとLinterをセットアップしたので、ここではDocstringとTypeHintsについて説明する。

## Docstring

DocstringにはGoogleスタイル, reStructuredTextスタイル、NumPyスタイルの3つがあります。WALCのでは[Googleスタイル](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)を使用します。

```python
def func(arg1, arg2, arg3):
  """Hello, func

  Lorem ipsum dolor sit amet,

  Args:
    arg1 (str): First argument
    arg2 (list[int]): Second argument
    arg3 (dict[str, int]): Third argument

  Returns:
    str or None: Return value

  Raises:
    ValueError: if arg1 is empty string.
  """
  ```

## TypeHints

Python3.6からTypeHintsという型アノテーションの機能が実装されました。Type Hintsを書くことで、コードの可読性が向上します。またPylanceのような静的解析ツールを使用することで、バグを事前に防ぐことができます。[Python公式ドキュメント](https://docs.python.org/3/library/typing.html)を参考に記述してください。

```python
def f(num1: int, my_float: float = 3.5) -> float:
    return num1 + my_float
```

### TypeHintsの注意点

Python3.9以上ではTypeHintsとして組み込み関数が使用できます。Python3.7, 3.8で開発を行う場合は互換性を担保するために[futureクラス](https://docs.python.org/3.8/library/__future__.html)を使いましょう。例えば、`typing.Union[int,float]`は`int|float`と書けます。また`typing.List[int]`も`list[int]`と書くことができます。

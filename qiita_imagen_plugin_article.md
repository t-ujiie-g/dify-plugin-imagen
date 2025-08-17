---
title: DifyでVertex AI Imagenが使える！カスタムツールを公開しました
tags:
  - Python
  - Dify
  - VertexAI
  - Imagen
  - 生成AI
---

## はじめに

こんにちは！

LLMアプリ開発プラットフォーム「Dify」で、Google Cloudの高性能な画像生成AI **Vertex AI Imagen** を簡単に利用できるカスタムツールを開発・公開しました。

**リポジトリはこちら (公開後にURLを更新してください)**
`https://github.com/your-username/dify-plugin-imagen`

この記事では、このプラグインの導入方法から使い方、そして開発の裏側までを解説します。Difyを使っている方、これから使いたいと思っている方は、ぜひこのプラグインを導入して、画像生成機能をあなたのLLMアプリケーションに組み込んでみてください。

## このプラグインで出来ること

Difyのワークフローに「Vertex AI Imagen」ツールを追加し、以下の機能を実現できます。

- **テキストからの画像生成**: プロンプト（指示文）に基づいて画像を生成します。
- **豊富なパラメータ設定**: 
    - **モデル選択**: `Imagen 3` や `Imagen 4` など、用途に応じたモデルを選択可能。
    - **アスペクト比**: `1:1`, `16:9`, `9:16`など、主要なアスペクト比に対応。
    - **画像枚数**: 一度に最大4枚までの画像を生成。

![DifyでImagenツールを使っている様子のスクリーンショット](ここにスクリーンショットを挿入)

## インストールと設定ガイド

導入はとても簡単です。3ステップで完了します。

### Step 1: GitHubリポジトリからツールをインポート

1. Difyにログインし、画面上部のメニューから「ツール」を選択します。
2. 「独自のツールを構築」をクリックします。
3. 「GitHubからインポート」を選択し、以下のURLを入力して「次へ」をクリックします。
   ```
   https://github.com/your-username/dify-plugin-imagen
   ```
   *(注: こちらはプレースホルダーです。ご自身の公開リポジトリURLに置き換えてください)*

### Step 2: 認証情報の設定

インポートが完了すると、ツールの設定画面が開きます。ここでVertex AIを利用するための認証情報を設定します。

1.  **GCPでサービスアカウントキーを発行**: まず、GCPコンソールでVertex AI APIへのアクセス権を持つサービスアカウントを作成し、キー（JSONファイル）をダウンロードしておきます。

2.  **サービスアカウントキーをBase64にエンコード**: ダウンロードしたJSONキーは、そのままではDifyに登録できません。以下のコマンドなどを使い、JSONファイルの中身を**Base64形式の文字列**に変換してください。

    ```bash
    # macOS の場合
    base64 -i /path/to/your/service-account-key.json

    # Linux の場合
    base64 -w 0 /path/to/your/service-account-key.json
    ```

3.  **Difyに認証情報を入力**: ツールの設定画面にある「認証情報」セクションに、以下の通り入力します。
    - `project_id`: あなたのGCPプロジェクトID
    - `location`: Vertex AIを使用するリージョン (例: `us-central1`)
    - `vertex_service_account_key`: 先ほど生成した**Base64形式の文字列**を貼り付けます。

    ![認証情報設定画面のスクリーンショット](ここにスクリーンショットを挿入)

4.  最後に「保存」をクリックします。

### Step 3: アプリケーションへの追加

あとは、このツールを使いたいアプリの「プロンプトエンジニアリング」画面を開き、「ツール」セクションから「vertexai-imagen」を追加すれば準備完了です。

## 使い方

チャット形式のアプリであれば、`@`でツールを呼び出すか、`/`をクリックしてツールリストから「Vertex AI Imagen」を選択します。

![ツールの使い方を示すスクリーンショット](ここにスクリーンショットを挿入)

プロンプトや各種パラメータを入力して実行すると、画像が生成されます。

---

## 技術的な解説（開発者向け）

ここからは、このプラグインがどのような仕組みで動いているのか、開発に興味がある方向けに解説します。

### プロジェクト構成

Difyのカスタムツールは、主に以下のファイルで構成されています。

- `manifest.yaml`: プラグイン全体のお約束事を定義します。
- `tools/vertexai-imagen.yaml`: DifyのUI（パラメータ入力欄など）を定義します。
- `tools/vertexai-imagen.py`: 画像生成のメインロジックを実装したPythonコードです。

### UIの定義 (YAML)

`tools/vertexai-imagen.yaml` で、ユーザーが操作するUIを定義します。

```yaml
# tools/vertexai-imagen.yaml
parameters:
  - name: prompt
    type: string
    required: true
    label: { ja_JP: テキストプロンプト }
    form: llm
  - name: model
    type: select # ドロップダウンリスト
    required: false
    label: { ja_JP: Imagen モデル }
    options:
      - { value: imagen-4.0-generate-preview-06-06, label: { ja_JP: "Imagen 4.0（高品質）" } }
      - { value: imagen-3.0-generate-001, label: { ja_JP: "Imagen 3.0（高品質）" } }
    default: imagen-4.0-generate-preview-06-06
    form: form
# ...
```

### メインロジック (Python)

`tools/vertexai-imagen.py` が処理の心臓部です。

```python
# tools/vertexai-imagen.py
class ImagenGenerateTool(Tool):
    def _invoke(self, tool_parameters: Dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        # 1. Difyに設定された認証情報を取得
        credentials = self.runtime.credentials
        project_id = credentials.get('project_id')
        service_account_key_b64 = credentials.get('vertex_service_account_key')

        # 2. Base64デコードしてGCPクレデンシャルを復元
        service_account_info = json.loads(base64.b64decode(service_account_key_b64))
        credentials_obj = service_account.Credentials.from_service_account_info(service_account_info)

        # 3. Vertex AIを初期化
        aiplatform.init(project=project_id, credentials=credentials_obj)

        # 4. ユーザーが入力したパラメータを取得
        prompt = tool_parameters.get('prompt')

        # 5. Imagen APIを呼び出し
        generation_model = ImageGenerationModel.from_pretrained(...)
        response = generation_model.generate_images(...)

        # 6. 結果を画像(Blob)としてDifyに返す
        for image in response.images:
            # ... (一時ファイルに保存してバイナリを読み込む処理)
            yield self.create_blob_message(blob=image_data, meta={'mime_type': 'image/png'})
```

特に重要なのが、**Base64エンコードされたサービスアカウントキーをデコードして認証を通す**部分です。これがDifyカスタムツールの開発における一つの勘所です。

## おわりに

Difyのカスタムツール機能を使うことで、既存のアプリケーションに強力な画像生成能力を簡単に追加できました。

このプラグインが、皆さんのLLMアプリ開発の助けになれば幸いです。ぜひ使ってみてください！

バグ報告や機能追加の要望、改善提案などがあれば、お気軽にGitHubのIssueやPull Requestをいただけると嬉しいです。

**リポジトリ:** `https://github.com/your-username/dify-plugin-imagen` (再掲)
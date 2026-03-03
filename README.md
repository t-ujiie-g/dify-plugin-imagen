# Vertex AI Imagen & Nano Banana Plugin for Dify

This is a custom tool for [Dify](https://dify.ai/) that allows you to generate images using Google Cloud's **Vertex AI Imagen** and **Gemini 3.1 Flash Image Preview (Nano Banana)**. Integrate high-quality image generation capabilities directly into your Dify AI applications.

## ✨ Features

- **Text-to-Image Generation**: Create images from text prompts.
- **Multi-Model Support**: 
  - **Imagen**: Choose between different Imagen versions like `Imagen 4` and `Imagen 3`.
  - **Gemini 3.1 Flash Image Preview (Nano Banana)**: Support for the latest highly efficient image generation model, including multi-modal interactions.
- **Aspect Ratio Control**: 
  - Imagen: Supports `1:1`, `16:9`, `9:16`, `4:3`, and `3:4`.
  - Nano Banana: Supports an extended range including `1:1`, `3:4`, `4:3`, `9:16`, `16:9`, `1:4`, `4:1`, `1:8`, and `8:1`.
- **Resolution Control**: Specify `0.5K`, `1K`, `2K`, or `4K` resolutions (for Nano Banana).
- **Safety Controls**: Configure safety filters for generated content (Imagen).
- **Image Input**: Support for image-to-image workflows by passing input images (Nano Banana).

## ⚙️ Requirements

- A Dify account (Cloud or Self-hosted).
- A Google Cloud Platform (GCP) project.
- Vertex AI API enabled in your GCP project.
- A GCP Service Account with permissions for Vertex AI, and its corresponding JSON key.

## Memo
- Package
```bash
dify plugin package ../dify-plugin-imagen
```

- Publish  
https://legacy-docs.dify.ai/ja-jp/plugins/publish-plugins/publish-plugin-on-personal-github-repo

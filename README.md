# Vertex AI Imagen Plugin for Dify

This is a custom tool for [Dify](https://dify.ai/) that allows you to generate images using Google Cloud's **Vertex AI Imagen**. Integrate high-quality image generation capabilities directly into your Dify AI applications.

## ✨ Features

- **Text-to-Image Generation**: Create images from text prompts.
- **Multiple Model Support**: Choose between different Imagen versions like `Imagen 4` and `Imagen 3`.
- **Aspect Ratio Control**: Supports `1:1`, `16:9`, `9:16`, `4:3`, and `3:4`.
- **Adjustable Image Count**: Generate 1 to 4 images per request.
- **Safety Controls**: Configure safety filters for generated content.

## ⚙️ Requirements

- A Dify account (Cloud or Self-hosted).
- A Google Cloud Platform (GCP) project.
- Vertex AI API enabled in your GCP project.
- A GCP Service Account with permissions for Vertex AI, and its corresponding JSON key.

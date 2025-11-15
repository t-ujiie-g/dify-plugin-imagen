import json
import base64
import tempfile
import os

from typing import Any, Dict
from PIL import Image
from collections.abc import Generator
from google.oauth2 import service_account
from google.genai import Client
from google.genai import types
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class NanoBananaGenerateTool(Tool):
    """
    Google Vertex AI Nano Banana tool for generating images from text prompts
    """

    def _invoke(self, tool_parameters: Dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke the Nano Banana image generation tool
        
        Args:
            tool_parameters: The parameters for the tool invocation
            
        Returns:
            List of tool invoke messages containing the generated images
        """
        try:
            # Get credentials and configuration
            credentials = self.runtime.credentials
            project_id = credentials.get('project_id')
            location = credentials.get('location', 'us-central1')
            service_account_key = credentials.get('vertex_service_account_key')
            SCOPES = [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language"
            ]
            
            if not project_id or not service_account_key:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": "Missing required credentials: project_id and service_account_key are required"}
                )]
            
            # Parse service account key
            try:
                service_account_info = json.loads(base64.b64decode(service_account_key))
                credentials_obj = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=SCOPES
                )
            except (json.JSONDecodeError, ValueError) as e:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": f"Invalid service account key format: {str(e)}"}
                )]
            
            # Initialize Vertex AI
            client =  Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials_obj
            )
            
            # Get tool parameters
            prompt = tool_parameters.get('prompt', '')
            image_input = tool_parameters.get('image', None)
            model_name = tool_parameters.get("model", "gemini-2.5-flash-image")

            if not prompt:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": "Prompt is required to generate images"}
                )]

            # Prepare contents
            contents = [prompt]
            if image_input:
                try:
                    image = Image.open(image_input)
                    contents.append(image)
                except Exception as e:
                    return [ToolInvokeMessage(
                        type=ToolInvokeMessage.MessageType.TEXT,
                        message={"text": f"Invalid image format: {str(e)}"}
                    )]

            # Generate images
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    )
                )
            except Exception as e:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": f"Error generating images: {str(e)}"}
                )]
            
            # Add each generated image
            for i, part in enumerate(response.parts):
                try:
                    if part.text is not None:
                        yield self.create_text_message(
                            text=part.text
                        )
                    if part.inline_data is not None:
                        # Get image using as_image() method
                        output_image = part.as_image()

                        # Create image message
                        yield self.create_blob_message(
                            blob=output_image.image_bytes,
                            meta={
                                'mime_type': 'image/png',
                                'filename': f'nanobanana_generated_{i+1}.png',
                                'alt': prompt,
                                'usage': 'generated_image'
                            }
                        )

                except Exception as e:
                    yield ToolInvokeMessage(
                        type=ToolInvokeMessage.MessageType.TEXT,
                        message={"text": f"Error processing image {i+1}: {str(e)}"}
                    )
            
        except Exception as e:
            yield ToolInvokeMessage(
                type=ToolInvokeMessage.MessageType.TEXT,
                message={"text": f"Unexpected error: {str(e)}"}
            )

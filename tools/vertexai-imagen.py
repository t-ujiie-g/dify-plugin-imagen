import json
import base64
import tempfile
import os

from typing import Any, Dict
from collections.abc import Generator
from google.oauth2 import service_account
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import aiplatform
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class ImagenGenerateTool(Tool):
    """
    Google Vertex AI Imagen tool for generating images from text prompts
    """

    def _invoke(self, tool_parameters: Dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke the Imagen image generation tool
        
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
            
            if not project_id or not service_account_key:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": "Missing required credentials: project_id and service_account_key are required"}
                )]
            
            # Parse service account key
            try:
                service_account_info = json.loads(base64.b64decode(service_account_key))
                credentials_obj = service_account.Credentials.from_service_account_info(service_account_info)
            except (json.JSONDecodeError, ValueError) as e:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": f"Invalid service account key format: {str(e)}"}
                )]
            
            # Initialize Vertex AI
            aiplatform.init(
                project=project_id,
                location=location,
                credentials=credentials_obj
            )
            
            # Get tool parameters
            prompt = tool_parameters.get('prompt', '')
            model_name = tool_parameters.get("model", "imagen-3.0-generate-001")
            aspect_ratio = tool_parameters.get('aspect_ratio', '1:1')
            number_of_images = tool_parameters.get('number_of_images', 1)
            safety_filter_level = tool_parameters.get('safety_filter_level', 'block_medium_and_above')
            
            if not prompt:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": "Prompt is required to generate images"}
                )]
            
            # Validate number of images
            if number_of_images < 1 or number_of_images > 4:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": "Number of images must be between 1 and 4"}
                )]
            
            # Initialize the Imagen model
            generation_model = ImageGenerationModel.from_pretrained(model_name)
            
            # Generate images
            try:
                response = generation_model.generate_images(
                    prompt=prompt,
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level=safety_filter_level,
                )
            except Exception as e:
                return [ToolInvokeMessage(
                    type=ToolInvokeMessage.MessageType.TEXT,
                    message={"text": f"Error generating images: {str(e)}"}
                )]
            
            # Add each generated image
            for i, image in enumerate(response.images):
                try:
                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                        image.save(temp_file.name)
                        
                        # Read image data
                        with open(temp_file.name, 'rb') as f:
                            image_data = f.read()
                        
                        # Clean up temporary file
                        os.unlink(temp_file.name)
                        
                        # Create image message
                        yield self.create_blob_message(
                            blob=image_data,
                            meta={
                                'mime_type': 'image/png',
                                'filename': f'imagen_generated_{i+1}.png',
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

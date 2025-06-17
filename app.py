import google.generativeai as genai
import gradio as gr
import yaml
from datetime import datetime
import uuid
import sys
import traceback

# Print version info for debugging
print(f"Python version: {sys.version}")
print(f"Gradio version: {gr.__version__}")

# Set up the model
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Note: Replace with your actual API key
genai.configure(api_key="AIzaSyBF8c8pxDa8Y6pUWcU-V9vvlFBDER4bY2E")
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def generate_changeset_id():
    """Generate a unique changeset ID with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"changeset_{timestamp}_{str(uuid.uuid4())[:8]}"

def generate_liquibase_changeset(user_prompt, existing_changelog, author_name, context_env):
    """Generate a new Liquibase changeset based on user description"""
    
    changeset_id = generate_changeset_id()
    
    input_prompt = f"""
You are a Liquibase expert. Generate a new changeset in YAML format based on the user's description.

USER REQUEST: {user_prompt}

EXISTING CHANGELOG:
{existing_changelog if existing_changelog and existing_changelog.strip() else "No existing changelog provided."}

AUTHOR: {author_name}
CONTEXT: {context_env}

Please generate a complete Liquibase changeset that:
1. Has a unique changeset ID: {changeset_id}
2. Includes proper rollback statements
3. Uses appropriate data types for the database
4. Includes proper constraints and indexes where needed
5. Follows Liquibase best practices
6. Is properly formatted YAML

Respond with ONLY the changeset YAML (not the full databaseChangeLog wrapper), starting with:

- changeSet:
    id: {changeset_id}
    author: {author_name}
    context: {context_env}
    comment: [Generated comment describing the change]
    changes:
      [Your generated changes here]
    rollback:
      [Your rollback statements here]

Examples of common operations:
- Add column: Use addColumn with proper type and constraints
- Drop column: Use dropColumn with rollback addColumn
- Create table: Use createTable with all columns and constraints
- Add index: Use createIndex with rollback dropIndex
- Modify column: Use modifyDataType or addNotNullConstraint
- Add foreign key: Use addForeignKeyConstraint

Make sure the changeset is production-ready and includes appropriate error handling.
"""

    try:
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        return f"Error generating changeset: {str(e)}"

def process_changeset_request(user_prompt, existing_changelog, author_name, context_env):
    """Main function to process user request and generate changeset"""
    
    try:
        if not user_prompt or not user_prompt.strip():
            return "Please provide a description of the database change you want to make."
        
        # Set defaults
        if not author_name or not author_name.strip():
            author_name = "developer"
        if not context_env or not context_env.strip():
            context_env = "dev"
        
        # Generate the changeset
        response = generate_liquibase_changeset(user_prompt, existing_changelog, author_name, context_env)
        
        return response
        
    except Exception as e:
        print(f"Error in process_changeset_request: {e}")
        traceback.print_exc()
        return f"Error processing request: {str(e)}"

# Alternative interface creation with explicit types
def create_interface_alternative():
    """Alternative interface creation method"""
    try:
        # Create inputs with explicit typing
        inputs = [
            gr.components.Textbox(
                placeholder="e.g., 'Add a column named email to the users table'",
                label="Describe Your Database Change",
                lines=3
            ),
            gr.components.Textbox(
                placeholder="Paste your current changelog.yml content here (optional)",
                label="Existing Changelog Content (Optional)",
                lines=5
            ),
            gr.components.Textbox(
                placeholder="your-username",
                label="Author Name",
                value="developer"
            ),
            gr.components.Textbox(
                placeholder="dev, test, prod",
                label="Context/Environment",
                value="dev"
            ),
        ]
        
        outputs = gr.components.Textbox(
            label="Generated Changeset",
            lines=20,
            show_copy_button=True
        )
        
        interface = gr.Interface(
            fn=process_changeset_request,
            inputs=inputs,
            outputs=outputs,
            title="🔄 Liquibase Changeset Generator",
            description="Describe your database change in plain English, and get a production-ready Liquibase changeset",
            examples=[
                [
                    "Add a column named 'email' of type VARCHAR(255) to the users table",
                    "",
                    "developer",
                    "dev"
                ],
                [
                    "Create a new table called 'orders' with id, user_id, product_id, and created_at columns",
                    "",
                    "developer",
                    "dev"
                ],
                [
                    "Create an index on the email column in the users table",
                    "",
                    "developer",
                    "dev"
                ],
            ],
            allow_flagging="never"
        )
        return interface
    except Exception as e:
        print(f"Error creating alternative interface: {e}")
        traceback.print_exc()
        return None

def create_interface():
    """Original interface creation method with error handling"""
    try:
        interface = gr.Interface(
            fn=process_changeset_request,
            inputs=[
                gr.Textbox(
                    placeholder="e.g., 'Add a column named email to the users table'",
                    label="Describe Your Database Change",
                    lines=3
                ),
                gr.Textbox(
                    placeholder="Paste your current changelog.yml content here (optional)",
                    label="Existing Changelog Content (Optional)",
                    lines=5
                ),
                gr.Textbox(
                    placeholder="your-username",
                    label="Author Name",
                    value="developer"
                ),
                gr.Textbox(
                    placeholder="dev, test, prod",
                    label="Context/Environment",
                    value="dev"
                ),
            ],
            outputs=gr.Textbox(
                label="Generated Changeset",
                lines=20,
                show_copy_button=True
            ),
            title="🔄 Liquibase Changeset Generator",
            description="Describe your database change in plain English, and get a production-ready Liquibase changeset",
            examples=[
                [
                    "Add a column named 'email' of type VARCHAR(255) to the users table",
                    "",
                    "developer",
                    "dev"
                ],
                [
                    "Create a new table called 'orders' with id, user_id, product_id, and created_at columns",
                    "",
                    "developer",
                    "dev"
                ],
                [
                    "Create an index on the email column in the users table",
                    "",
                    "harness",
                    "dev"
                ],
            ],
            allow_flagging="never"
        )
        return interface
    except Exception as e:
        print(f"Error creating original interface: {e}")
        traceback.print_exc()
        return create_interface_alternative()

if __name__ == "__main__":
    try:
        demo = create_interface()
        if demo is not None:
            demo.launch(
                debug=True,
                share=True,
                server_name="0.0.0.0",
                server_port=7860
            )
        else:
            print("Failed to create interface")
    except Exception as e:
        print(f"Error launching application: {e}")
        traceback.print_exc()
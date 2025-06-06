import yaml
import os

class PromptTemplateLoader:
    def __init__(self, yaml_path="/scratch/work/zhangl9/genaid/TestBed/adherence/prompt_templates.yaml"):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f)

    def get(self, template_name, **kwargs):
        """Return the formatted prompt for a given template"""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found.")
        
        template = self.templates[template_name]["prompt"]
        return template.format(**kwargs)

    def list_templates(self):
        return list(self.templates.keys())

if __name__ == "__main__":
    loader = PromptTemplateLoader()

    # Example data
    note_text = "Potilaalla on yskää ja kuumetta viiden päivän ajan."  # "The patient has had a cough and fever for five days."
    guideline_chunks = "- Älä määrää antibiootteja tavalliseen viruksen aiheuttamaan yskään."  # "- Do not prescribe antibiotics for a common viral cough."

    # Get formatted prompt
    prompt = loader.get("adherence_check", note_text=note_text, guideline_chunks=guideline_chunks)
    print(prompt)

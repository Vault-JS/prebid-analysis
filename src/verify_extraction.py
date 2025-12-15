
import yaml
from pathlib import Path

def verify_extraction():
    # Paths
    base_dir = Path(__file__).parent.parent
    prompts_path = base_dir / "prompts.yaml"
    modules_dir = base_dir / "Prebid.js" / "modules"

    # Load Prompt
    print(f"Loading prompts from {prompts_path}")
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)

    instruction_template = prompts["extract_prebid_vendor"]["instructions"]

    # Load Sample Adapter (AppNexus)
    adapter_name = "appnexusBidAdapter.js"
    adapter_path = modules_dir / adapter_name

    if not adapter_path.exists():
        print(f"Error: Adapter {adapter_name} not found in {modules_dir}")
        return

    print(f"Loading adapter code from {adapter_path}")
    with open(adapter_path, "r") as f:
        js_content = f.read()

    # Construct Prompt (Simulation)
    # This is how the prompt matches the input_schema in prompts.yaml intuitively
    prompt = f"""
{instruction_template}

---
Input Data:
Vendor Name: appnexus
Module Type: BidAdapter
File Content ({adapter_name}):
{js_content[:2000]}
... [truncated for brevity in verification] ...
"""

    output_path = base_dir / "verification_prompt.txt"
    with open(output_path, "w") as f:
        f.write(prompt)

    print(f"Verification prompt saved to {output_path}")
    print("Success! Prompt construction verified.")

if __name__ == "__main__":
    verify_extraction()

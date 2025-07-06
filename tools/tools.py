import os
import json

def save_json(input_data, save_dir="./saved/results.json") -> None:
    """
    Save input_data to a JSON file, appending to existing data if the file exists.
    
    Args:
        input_data: Data to save (expected to be a JSON-serializable object, e.g., list or dict).
        save_dir: Path to the JSON file (default: "./saved/results.json").
    
    Raises:
        ValueError: If input_data is not valid JSON-serializable.
        OSError: If there are issues with file operations.
    """
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    try:
        json.dumps(input_data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Input data is not valid JSON: {str(e)}")
    
    if os.path.exists(save_dir):
        try:
            with open(save_dir, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Existing file {save_dir} contains invalid JSON")
            
            if not isinstance(existing_data, list):
                raise ValueError(f"Existing file {save_dir} does not contain a JSON array")
            
            if isinstance(input_data, dict):
                input_data = [input_data]
            elif not isinstance(input_data, list):
                raise ValueError("Input data must be a dict or list for appending")
            
            existing_data.extend(input_data)
            data_to_save = existing_data
        except OSError as e:
            raise OSError(f"Error reading file {save_dir}: {str(e)}")
    else:

        if isinstance(input_data, dict):
            data_to_save = [input_data]
        elif isinstance(input_data, list):
            data_to_save = input_data
        else:
            raise ValueError("Input data must be a dict or list for saving")
    
    try:
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
    except OSError as e:
        raise OSError(f"Error writing to file {save_dir}: {str(e)}")
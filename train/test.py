import json
with open("../config.json") as f:
        data = json.load(f)
print(data["facial_image_size"]["image_height"])
print(data["facial_image_size"]["image_width"])

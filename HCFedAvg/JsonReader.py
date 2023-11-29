import json
with open('package.json',encoding='utf-8') as f:
    superHeroSquad = json.load(f)
print(type(superHeroSquad))  # Output: dict
print(superHeroSquad["Config_1"]["Description"])
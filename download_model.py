#!/usr/bin/env python3

#Helper script to download specific transformer models and save them locally
import os
from sentence_transformers import SentenceTransformer


#modelName = "all-MiniLM-L6-v2"
modelName = "paraphrase-multilingual-MiniLM-L12-v2"
modelPath = "./Models/"+modelName

print(f"Given model path is: {modelPath}")

if os.path.isdir(modelPath):  # Path exists 
	if not os.listdir(modelPath):   # and is empty
		model = SentenceTransformer(modelName)
		model.save(modelPath)
		print(f"Model downloaded to {modelPath}")
	else: # perhaps model already exists   
		print("Directory is not empty.  Nothing downloaded.")
else:
	print("Given directory doesn't exist")

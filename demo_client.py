#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo client to talk to the Briefly summarizer service.
Start the service with:
    fastapi run briefly.py
"""

from talk_briefly import BrieflyClient # Import the Briefly client module

bc = BrieflyClient("http://localhost:8000")  # Instantiate the Briefly client

print(f"Service is available: {bc.is_okay()}")  # Check if the service is available

old_opt = bc.get_options()  # Get current options and print
print(f"Current options:{old_opt}")

# Modify any current options that you choose
new_opt = old_opt.copy()
new_opt.update({"merge_threshold":0.8,"passes":1})

print(f"Options set as: {bc.set_options(new_opt)}")  # Set and print new options

with open("./Text/mayon_volcano.txt","r") as f:  # Read a file to summarize
    all_lines = f.read()
    
print(f"Summarized text:\n{bc.get_summary(all_lines)}") # Get and print the summary



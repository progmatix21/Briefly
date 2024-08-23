# -*- coding: utf-8 -*-
"""
# Briefly:talk_briefly.py - client library module for Briefly REST summarizer service
# Copyright (C) 2024  Atrij Talgery: github.com/progmatix21
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://www.gnu.org/licenses/agpl.txt
# https://spdx.org/licenses/AGPL-3.0-or-later.html
"""

import requests
import abc


class NotImplementedError(BaseException):
    # Raised when an unimplemented base class method is called
    pass

class ServiceUnavailableError(BaseException):
    # Raised when a service is not available
    pass

class BrieflyAPI(object):
    """Abstract base class for BrieflyAPI."""
    
    __class__ = abc.ABCMeta


    def set_options(self,options):
        raise NotImplementedError
        
    def get_options(self):
        raise NotImplementedError
        
    def get_summary(self,file_lines):
        raise NotImplementedError
        
    def is_okay(self):
        raise NotImplementedError
        
        
class BrieflyClient(BrieflyAPI):
    """Concrete class implementing BrieflyAPI interface."""
    
    top_endpoint = "/"
    options_endpoint = "/options"
    summary_endpoint = "/summary"
    
    def __init__(self,base_api_url):
        self.url = base_api_url

    
    def get_options(self):
        """Get the options from the summarizer service via REST API and return it."""
        if self.is_okay():
            response = requests.get(self.url+BrieflyClient.options_endpoint).json()
            return str(response)
        else:
            return {}

    
    def set_options(self, options):
        """Set the options for the summarizer service REST API and return updated options."""
        
        response = requests.put(self.url+BrieflyClient.options_endpoint,json=options)
        
        return str(response.json())


    def get_summary(self,file_lines):
        """Create a summary resource and get it."""
        # Input:blob of text in file_lines
        return(requests.post(self.url+BrieflyClient.summary_endpoint,
            json={"text":f"{file_lines}"}).json()["text"])

    
    def is_okay(self):
        """Check if connection is established."""
        
        response = requests.get(self.url+BrieflyClient.top_endpoint)
        if str(response.status_code) == "200":
            return True
        else:
            raise ServiceUnavailableError
            

import requests
import json
import abc
import sys

'''
print(requests.get("http://127.0.0.1:8000/").json())
print(requests.get("http://127.0.0.1:8000/").json()['message'])

print("Current options are: ")
print(requests.get("http://127.0.0.1:8000/options").json())

new_opt = {'filename': "", 'merge_threshold': 0.9, 'min_word_count': 2, 'passes': 1, 'summary_size': 1, 'include_context': False, 'verbose': False}

print("New options are: ")
print(json.dumps(new_opt))

response = requests.put("http://localhost:8000/options",json=new_opt)

print("Options have been set as: ")
print(response.json())

print("Get summary")
with open("./Text/mayon_volcano.txt","r") as f:
    for file_lines in f:
        file_lines = f.readlines()
        #print(file_lines)


print(requests.post("http://127.0.0.1:8000/summary",
    json={"text":f"{file_lines}"}).json()["text"])

'''
class NotImplemented(BaseException):
    # Raised when an unimplemented base class method is called
    pass

class ServiceUnavailableError(BaseException):
    # Raised when a service is not available
    pass

class BrieflyAPI(object):
    
    __class__ = abc.ABCMeta

    

    def set_options(self,options):
        raise NotImplemented
        
    def get_options(self):
        raise NotImplemented
        
    def get_summary(self):
        raise NotImplemented
        
    def is_okay(self):
        raise NotImplemented
        
class VirtualSummarizer(BrieflyAPI):
    
    top_endpoint = "/"
    options_endpoint = "/options"
    summary_endpoint = "/summary"
    
    def __init__(self,base_api_url):
        self.url = base_api_url
    
    def get_options(self):
        '''Get the options from the REST API and return it.'''
        if self.is_okay():
            response = requests.get(self.url+VirtualSummarizer.options_endpoint).json()
            return str(response)
        else:
            return {}
    
    def set_options(self, options):
    
        #print("Old options are: ")
        #print(json.dumps(new_opt))
        #old_response = json.dumps(options)
        #old_response = requests.get(self.url+VirtualSummarizer.options_endpoint).json()
        #print(old_response)
        
        #options = new_opt
        response = requests.put(self.url+VirtualSummarizer.options_endpoint,json=options)
        
        print("Options have been set as: ")
        return str(response.json())

    def get_summary(self,file_lines):
        print("Get summary")
        #with open(filepath,"r") as f:
        #    file_lines = f.read()

        return(requests.post(self.url+VirtualSummarizer.summary_endpoint,
            json={"text":f"{file_lines}"}).json()["text"])

    
    def is_okay(self):
        '''Check if connection is established.'''
        response = requests.get(self.url+VirtualSummarizer.top_endpoint)
        if str(response.status_code) == "200":
            return True
        else:
            raise ServiceUnavailableError
            
'''
if __name__ == "__main__":

    url = "http://127.0.0.1:8000"
    new_opt = {'filename': "", 'merge_threshold': 0.55, 'min_word_count': 2,
        'passes': 2, 'summary_size': 1, 'include_context': False, 'verbose': False}
    
    
    filepaths = ["./Text/mayon_volcano.txt","./Text/aeon_row5.txt","./Text/ukraine_dam.txt","./Text/hindi1.txt"]
    mysummarizer = VirtualSummarizer(url)
    try:
        print(mysummarizer.set_options(new_opt))
    except NotImplemented:
        print("Exception: calling unimplemented base class method.")
    
    for filepath in filepaths:
        try:
            print(filepath)
            print(mysummarizer.get_summary(filepath))
        except NotImplemented:
            print("Exception: calling unimplemented base class metnod.")
'''
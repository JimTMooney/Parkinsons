import requests, zipfile, io, os

"""

Retrieves a zip file from a given url and writes to the data directory

args:

url --> url to retreive the zip file from
file_type --> subdirectory name to extract zip file into

"""
def write_url(url, file_type):
    print('fetching ' + url)
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    write_location = os.path.join('../../data', file_type)
    z.extractall(write_location)
    
    
"""
Retrieves handpd spiral and meander images for healthy individuals and patients
"""
def retreive_images():
    base_url = "http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/New"
    append_str = ["Healthy/Healthy", "Patients/Patient"]
    append_str = [append_str[0] + 'Spiral', append_str[0] + 'Meander', 
                  append_str[1] + 'Spiral', append_str[1] + 'Meander']
    full_urls = [base_url + app_string + '.zip' for app_string in append_str]
    
    for url in full_urls:
        write_url(url, 'Images')
      
    
"""
Retrieves handpd signal dataset
"""
def retreive_signals():
    base_url = "http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/New"
    end_string = "/Signal.zip"
    middle_strings = ["Healthy", "Patients"]
    
    full_urls = [base_url + middle + end_string for middle in middle_strings]
    
    for idx, url in enumerate(full_urls):
        file_type = os.path.join("Signals", middle_strings[idx])
        write_url(url, file_type)
    
        

"""
If run as main, this file retrieves all necessary data for this project.

To run, navigate to this file's location in the terminal. Then run

python3 make_dataset.py

"""
if __name__ == "__main__":
    retreive_images()
    retreive_signals()
# Project Documentation

Template for the .env file

```bash
# uri_type
uri_folder = 'folder'
uri_file = 'file'

# datastore_uri
# for downloading raw csv files
# go to ml.azure.com -> Data -> Datastores -> <name> -> browse -> click ... then Copy URI
data1_uri = <datastore uri>
data2_uri = <datastore uri>

# For downloading
# Use connect string of key1 as key was used to create the datastore
connect_str = <connect string>
container_name = <container name>

# Names blobs to be downloaded
blob1 = '<path>\\file1.csv'
blob2 = '<path>\\file2.csv'
```

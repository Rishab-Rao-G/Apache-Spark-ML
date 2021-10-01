import wfdb
# It will take some time to download the dataset
# The dataset will be downloaded inside the directory 'data'
def download():
    wfdb.dl_database('mitdb', dl_dir = 'data')

if __name__ == "__main__":
    download()

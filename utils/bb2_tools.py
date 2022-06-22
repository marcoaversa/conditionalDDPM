import os
import shutil

from b2sdk.v1 import *

def get_b2_bucket():
    bucket_name = 'perturbed-minds'
    application_key_id = '003d6b042de536a0000000008'
    application_key = 'K003HMNxnoa91Dy9c0V8JVCKNUnwR9U'
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', application_key_id, application_key)
    bucket = b2_api.get_bucket_by_name(bucket_name)
    return bucket

def b2_list_files(folder=''):
    bucket = get_b2_bucket()
    for file_info, _ in bucket.ls(folder, show_versions=False):
        print(file_info.file_name)

def b2_download_folder(b2_dir, local_dir, force_download=False, mirror_folder=True):
    """Downloads a folder from the b2 bucket and optionally cleans
    up files that are no longer on the server
    Args:
        b2_dir (str): path to folder on the b2 server
        local_dir (str): path to folder on the local machine
        force_download (bool, optional): force the download, if set to `False`, 
            files with matching names on the local machine will be skipped
        mirror_folder (bool, optional): if set to `True`, files that are found in
            the local directory, but are not on the server will be deleted
    """
    bucket = get_b2_bucket()

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    elif not force_download:
        return

    download_files = [file_info.file_name.split(b2_dir + '/')[-1]
                      for file_info, _ in bucket.ls(b2_dir, show_versions=False)]

    for file_name in download_files:
        if file_name.endswith('/.bzEmpty'):  # subdirectory, download recursively
            subdir = file_name.replace('/.bzEmpty', '')
            if len(subdir) > 0:
                b2_subdir = os.path.join(b2_dir, subdir)
                local_subdir = os.path.join(local_dir, subdir)
                if b2_subdir != b2_dir:
                    b2_download_folder(b2_subdir, local_subdir, force_download=force_download,
                                       mirror_folder=mirror_folder)
        else:   # file
            b2_file = os.path.join(b2_dir, file_name)
            local_file = os.path.join(local_dir, file_name)
            if not os.path.exists(local_file) or force_download:
                print(f"downloading b2://{b2_file} -> {local_file}")
                bucket.download_file_by_name(b2_file, DownloadDestLocalFile(local_file))

    if mirror_folder:   # remove all files that are not on the b2 server anymore
        for i, file in enumerate(download_files):
            if file.endswith('/.bzEmpty'):  # subdirectory, download recursively
                download_files[i] = file.replace('/.bzEmpty', '')
        for file_name in os.listdir(local_dir):
            if file_name not in download_files:
                local_file = os.path.join(local_dir, file_name)
                print(f"deleting {local_file}")
                if os.path.isdir(local_file):
                    shutil.rmtree(local_file)
                else:
                    os.remove(local_file)   
import os

def getProjectPath(last_folder_name):
  '''
    end without /
  '''
  cur_file = os.path.abspath(__file__)
  start_index = cur_file.find(last_folder_name)
  pro_path = cur_file[:start_index+11]
  return  pro_path



if __name__ == "__main__":
  save_dir = getProjectPath("auto_filter")+ "/workdata/"
  pass

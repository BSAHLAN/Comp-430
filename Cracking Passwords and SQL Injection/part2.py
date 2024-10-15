import hashlib
import numpy as np

#reading files
def read_stolen():
    
    user_dict = {}
    
    with open("salty-digitalcorp.txt", 'r') as file:
        for line in file:

            row = line.strip().split(',')
        
            if(row[1] == "salt"):
                continue
            
            username = row[0]
            salt = row[1]
            hash = row[2]
            user_dict[username] = [salt, hash] #first element is salt, second element is hashed password
            
    return user_dict


def read_rockyou():
    
    stolen_list = []
    with open("rockyou.txt", 'r') as file:
        for line in file:
            pwd = line.strip()
            stolen_list.append(pwd)

    return stolen_list

  
#performing the dictionary attack
def dict_attack(users:dict, passwords:list):
    
    attack_dict = {}
    for user in users:
        user_info = users[user]
        for pwd in passwords:
             user_salt = user_info[0]
             hashed_user_pwd = user_info[1]
             prepended = user_salt + pwd
             appended = pwd + user_salt
             
             pre_hash = hashlib.sha512(prepended.encode("utf-8")).hexdigest()
             app_hash = hashlib.sha512(appended.encode("utf-8")).hexdigest()
             
             if((hashed_user_pwd == pre_hash) or (hashed_user_pwd == app_hash)):
                 attack_dict[user] = pwd
                 break
            
    return attack_dict
            


if __name__ == "__main__":
    
    stolen_dict = read_stolen()
    breach_list = read_rockyou()
    result = dict_attack(stolen_dict, breach_list)
    
    print(result)
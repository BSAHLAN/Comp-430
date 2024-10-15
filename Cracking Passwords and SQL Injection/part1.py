import hashlib
import numpy as np

#reading files
def read_stolen():
    
    user_dict = {}
    
    with open("digitalcorp.txt", 'r') as file:
        for line in file:

            row = line.strip().split(',')
        
            if(row[1] == "hash_of_password"):
                continue
            
            username = row[0]
            hash = row[1]
            user_dict[username] = hash
            
    return user_dict

def read_rockyou():
    
    stolen_dict = {}
    with open("rockyou.txt", 'r') as file:
        for line in file:
            pwd = line.strip()
            stolen_dict[pwd] = hashlib.sha512(pwd.encode("utf-8")).hexdigest()

    return stolen_dict

#performing the dictionary attack
def dict_attack(users:dict, hashes:dict):
    
    attack_dict = {}
    for user in users:
        hashed_pwd = users[user]
        for pwd in hashes:
            if(hashed_pwd == hashes[pwd]):
                attack_dict[user] = pwd
                break
    return attack_dict
            


if __name__ == "__main__":
    
    stolen_dict = read_stolen()
    breach_dict = read_rockyou()

    result = dict_attack(stolen_dict, breach_dict)

    print(result)





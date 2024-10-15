import hashlib
import numpy as np
import binascii
#from backports.pbkdf2 import pbkdf2_hmac
import hmac

#reading files
def read_stolen():
    
    user_dict = {}
    
    with open("keystreching-digitalcorp.txt", 'r') as file:
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

def calculateHash(element_list:list):
    
    new_str = ""
    
    for elem in element_list:
        if(isinstance(elem, list)):
           new_str += elem[0]
        
        else:
            new_str += elem
            
    #element_str = "".join(element_list)
    
    return hashlib.sha512(new_str.encode("utf-8")).hexdigest()
 
   
#performing the dictionary attack
def dict_attack(users:dict, passwords:list):
    
    attack_dict = {}
    max_key_iterations = 2000
    min_key_iterations = 1
    is_found = False #a checker for checking if the number of iterations and combination found 
    combination = 0
    real_iteration = 0
    
    for user in users:
    
        user_info = users[user]
        user_salt = user_info[0]
        hashed_user_pwd = user_info[1]
        
        for pwd in passwords:
             
             if(user in attack_dict.keys()):
                 break
              
             result_list = [""]
             
             l1 = [result_list, pwd, user_salt]
             l2 = [result_list, user_salt, pwd]
             l3 = [user_salt, pwd, result_list]
             l4 = [user_salt, result_list, pwd]
             l5 = [pwd, user_salt, result_list]
             l6 = [pwd, result_list, user_salt]
             
             combinations = [l1, l2, l3, l4, l5, l6]
             
             if(not is_found):
                   for i in range(len(combinations)):
                       result_list[0] = ""
                       for j in range(max_key_iterations):
                           hash = calculateHash(combinations[i])
                                                       
                           if(hash == hashed_user_pwd):
                               attack_dict[user] = pwd
                               combination = i
                               real_iteration = j+1
                               is_found = True
                               break
                               
                            
                           else:
                               result_list[0] = hash 
                        
                       if(is_found):
                            break
                                            
             elif(is_found):
                 hash = 0
                 for i in range(real_iteration):
                     hash = calculateHash(combinations[combination])
                     result_list[0] = hash
                 
                 if(hash == hashed_user_pwd):
                     attack_dict[user] = pwd
                     
    return attack_dict


if __name__ == "__main__":
    
    stolen_dict = read_stolen()
    breach_list = read_rockyou()
    result = dict_attack(stolen_dict, breach_list)
    
    print(result)
    
    
    
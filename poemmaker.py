import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import pickle  
import random
import os
import copy
    
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, indices = torch.topk(logits, k)
    #print(indices)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits), min_values
 
#sound_file = open("sound.txt", "r")
#rhymeset_file = open("rhymesets.txt", "w")
#wordlist = []
#numlist = []
#soundlist = []
#for line in sound_file:
#    word, rest = line.split(":")
#    num, sound = rest.split(";")
#    wordlist.append(word)
#    soundlist.append(sound)
#    numlist.append(int(num))
#rhymesets=[]
#for num, sound, word in zip(numlist, soundlist, wordlist):
#    rhymeset=[num]
#    for num2, sound2, word2 in zip(numlist, soundlist, wordlist):
#        if sound == sound2:
#            rhymeset.append(num2)
#    rhymesets.append(rhymeset)
#pickle.dump( rhymesets, open( "rhymesets.p", "wb" ) )
#os.system("$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding")
rhymesets = pickle.load( open( "rhymesets.p", "rb" ) )
rhyme_mat = [0] * 50257
big_rhymesets = [[0]] * 50257
new_rhymesets = []
for rhymeset in rhymesets:
    if len(rhymeset)>4:
        new_rhymesets.append(rhymeset)
        big_rhymesets[rhymeset[1]]=rhymeset
        rhyme_mat[rhymeset[1]] = 1
rhymesets = new_rhymesets  
boring_rhymes = [" a"," an"," it"," is"," as"," at"," was"," of"," at"," that",
                 " has"," your"," my"," his"," their"," on"," for"," its"," to",
                 " from"," if"," ur"," re"," our"," un"," dis"," diss"," mis",
                 " wat"," com"," comm"," psych"," lol"," vis"," al"," los"," pol",
                 " bis"," up", " la"," sa"," ha"," mah",
                 " b"," c"," d"," e"," f"," g"," h"," i"," j"," k"," l"," m",
                 " n"," o"," p"," q"," r"," s"," t"," u"," v"," w"," x"," y"," z"]

print("rhymes loaded")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')   
bad_rhymes = []
for rhyme in boring_rhymes:
    token = tokenizer.encode(rhyme,add_prefix_space=True )
    bad_rhymes.append(token)
#model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("poetry")
#model = GPT2LMHeadModel.from_pretrained("gpt2")

print("model loaded")

with torch.no_grad():
    while True:
        prompt = input("starting prompt: ")
        #line_length = random.randrange(1,9) 
        line_length =int(input("line length: "))
        lines =int(input("lines: "))
        
        indexed_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(indexed_tokens).unsqueeze(0) # Batch size 1
        inputs = {'input_ids': input_ids}
        past = None
        next_token = []
        token_list = []
        penultimate_logits = []
        rhyme_with_me = 10
        
        jj=1
        tries=0
        # generate this many tokens:
        while jj < line_length*lines+1:
            return_flag = 0
            logits, past = model(**inputs, past=past)
            logits = logits[:, -1, :] 
            # prevent <|endoftext|>, and line breaks in the middle of a line 
            logits[0,50256]=-1e10
            logits[0,27]=-1e10 
            logits[0,1279]=-1e10
            logits[0,198]=-1e10
            logits[0,628]=-1e10
 
            # the rhyming word   
            if (jj % (line_length*2)) == (line_length*2)-3:
                cue = rhyme_with_me
                for rhyme in bad_rhymes:
                    logits[0,rhyme]=-1e10
                # no rhyming a word with itself
                logits[0,cue]=-1e10
                # only allow words that rhyme with cue
                for t in range(0,50257):
                    if t in big_rhymesets[cue]:
                        pass
                    else:
                        logits[0,t] = -1e10
                best_val, best_logit = torch.topk(logits, k=1)
                # reject really bad rhymes
                if tries<30:
                    if best_val[0][0].item()<-.5:
                        #print(int(best_val[0][0].item()),end=" ")
                        jj=jj-1
                        token_list.pop()
                        logits = copy.copy(penultimate_logits)
                        logits[0,token_list[-1]]=-1e10
                        tries = tries + 1
                    else:
                        tries = 0
                else:
                    print("*",end="")
                    tries = 0
            
            #the word to be rhymed with
            else:
                if (jj % line_length) == line_length-3:
                    # this flag says the token generated for this word becomes the cue
                    return_flag = 1
                    for rhyme in bad_rhymes:
                        logits[0,rhyme]=-1e10
                    # get rid of every token except those that have some words that rhyme with them 
                    for t in range(0,50256):
                        if rhyme_mat[t]==0:
                            logits[0,t]=-1e10
                best_val, best_logit = torch.topk(logits, k=1)
             
            #penultimate word in a line
            if (jj % line_length) == line_length-4:    
                logits[0,13]=-1e10
                logits[0,11]=-1e10
                penultimate_logits = copy.copy(logits)
            
            #punctuation to end a line                
            if (jj % line_length) == line_length-2: 
                values=[]
                #keep the probabilities for these punctuation tokens, set everything else to -1e10
                line_ends = [",",".",";","!",":","--"]
                for punct in line_ends:
                    token = tokenizer.encode(punct)
                    values.append(logits[0,token[0]].item())
                #allow a line to end with a blank space, too
                space_one = logits[0,220].item()
                logits[0,:]=-1e10
                for punct, value in zip(line_ends,values):
                    token = tokenizer.encode(punct)
                    logits[0,token[0]]= value
                logits[0,220]= space_one
            
            #the carriage return ending a line is the only allowable token
            if (jj % (line_length)) == line_length-1:  
                logits[0,:]=-1e10
                logits[0,198]=100
                for token in token_list:
                    print(tokenizer.decode(token),end="")
                token_list = []

            #reduce the chaos a bit
            logits, values = top_k_logits(logits, k=30)
            log_probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(log_probs, num_samples=1)
            token_list.append(next_token)
            #print(tokenizer.decode(next_token),end="")
            #set the cue to the token that's just been generated
            if return_flag == 1:
                rhyme_with_me = next_token[0][0].item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            inputs = {'input_ids': next_token}
            jj=jj+1
        print("\n----------------")
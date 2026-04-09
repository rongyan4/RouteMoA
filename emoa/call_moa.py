from openai import OpenAI
import time
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
client = OpenAI(
    api_key='YOUR_API_KEY',  # API key not needed
    base_url = "http://localhost:10666/v1"
)

payload = {
    "model": "moa_mixed_models",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        #{"role": "user", "content": "What are the names of some famous actors that started their careers on Broadway?"}

        #{"role": "user", "content": "Here is a Chinese Gaokao biology multiple-choice question. Please choose the correct answer.\nWhich statement about how organisms respond to stimuli is incorrect?\n(A) Viral infection -> human T cells secrete specific antibodies -> clear virus\n(B) Temperature drops -> mammal thermoregulation center activated -> stable body temp\n(C) High sugar intake -> increased insulin secretion -> blood sugar drops\n(D) Unilateral light -> auxin redistribution -> phototropic bending\nThe answer is:"}
        #{"role": "user", "content": 'A housefly sits on the outer edge of a rotating circular ceiling fan with a diameter of 6 feet. The fan rotates constantly at a rate of 20 revolutions per minute. How many minutes had the housefly been on the fan during the time it took to travel $19{,}404\\pi$ feet? Express your answer to the nearest whole number.\nPlease reason step by step, and put your final answer within \\boxed{}.'}

        #{"role": "user","content": "A beetle walks along the outer edge of a circular garden with a diameter of 10 feet. The garden’s sprinkler rotates constantly at a rate of 15 revolutions per minute. How many minutes had the beetle been walking during the time it took to travel $13{,}500\\pi$ feet? Express your answer to the nearest whole number.\\nPlease reason step by step, and put your final answer within \\boxed{}."}
        #{"role": "user", "content": "Translate 'Beauty is in the eye of the beholder' into Chinese"}
        #{"role": "user", "content": "Please translate 'China is a vast country with a long history and abundant resources' into English" }
        #{"role": "user", "content": "Please recommend some famous sites in New York and Shanghai"}
        #{"role": "user", "content": "Make a detailed introduction to Shanghai Jiao Tong University,University of Michigen and their relations "}
        # ARC gold C
        # {"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nAn astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\n\nA. Planetary density will decrease.\nB. Planetary years will become longer.\nC. Planetary days will become shorter.\nD. Planetary gravity will become stronger."}
        # ARC gold B
        {"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA group of engineers wanted to know how different building designs would respond during an earthquake. They made several models of buildings and tested each for its ability to withstand earthquake conditions. Which will most likely result from testing different building designs?\n\nA. buildings will be built faster\nB. buildings will be made safer\nC. building designs will look nicer\nD. building materials will be cheaper"}
        # biomed gold A
        #{"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA patient suffers a broken neck with damage to the spinal cord at the level of the sixth cervical vertebra.\n\nA) They will be unable to breathe without life support.\nB) They will only be able to breathe quietly.\nC) It is impossible to predict an effect on breathing.\nD) Breathing will be unaffected.\n"}
        
        # ---------------------------------------------------- Jue 22-----------------------------------------------------------------------------------------------
        # arc works properly

        #{"role": "user", "content":"Mosquito fish found on the islands of the Bahamas live in various isolated freshwater ponds that were once a single body of water. When several male and female mosquito fish are taken from two isolated ponds and placed into a single pond, the breeding preference of each mosquito fish is for fish from its own original pond. Which of these most likely resulted in this breeding preference?\n\nA. Availability of food influenced the breeding preferences of the fish.\nB. Competition for a suitable mate influenced the breeding preferences.\nC. Predators in the pond influenced the breeding preferences of the fish.\nD. Speciation due to reproductive isolation influenced the breeding preferences."}
        #{"role": "user", "content":"Geologists have identified seven major tectonic plates at the surface of Earth. Which evidence best indicates that tectonic plates collide?\n\nA. Wind erodes surface rock formations.\nB. Fossils date back thousands of years.\nC. Small rocks are left behind when glaciers retreat.\nD. Older layers of rock are located above newer layers of rock."}
        
        # math
        #{"role": "user", "content":"The expression $x^2 - 16x + 60$ can be written in the form $(x - a)(x - b)$, where $a$ and $b$ are both nonnegative integers and $a > b$. What is the value of $3b - a$?"}
        #{"role": "user", "content":"Let $\\textrm{A}$ be a digit. If the 7-digit number $353808\\textrm{A}$ is divisible by 2, 3, 4, 5, 6, 8, and 9, then what is $\\textrm{A}$?"}

        # gsm8k
        #{"role": "user", "content": "A cat spends its time hunting birds.  The cat catches 8 birds during the day and twice this many at night. In total, how many birds did the cat catch?"}
        #{"role": "user", "content": "A parking garage of a mall is four stories tall. On the first level, there are 90 parking spaces. The second level has 8 more parking spaces than on the first level, and there are 12 more available parking spaces on the third level than on the second level. The fourth level has 9 fewer parking spaces than the third level. If 100 cars are already parked, how many cars can be accommodated by the parking garage?"}

        # race
        #{"role": "user", "content": "Article: Mother-of-three Carmen Blake called her midwife to ask for an ambulance when she _ unexpectedly with her fourth child.\nBut the 27-year-old claims she was refused an ambulance and told to walk the 100m from her house in Leicester to the city's nearby Royal Infirmary .\nHer daughter Mariah was delivered on a pavement outside the hospital by a passer-by, just before ambulance crews arrived.\nMs Blake said she started going into labor at about 7:15 am on Sunday, August 2. She said, \"I phoned up the Royal Infirmary, it's just across the road.\n\"I went into the bath and realized she was gong to come quickly. I didn't think I'd be able to make it out of the bath, so I phoned the maternity  ward back and told them to get an ambulance out.\"\nThey said they were not sending an ambulance and told me I had had nine months to sort out a lift.\nExperienced mother MS Blake today said she knew she had to get herself out of the bath and try to get to the hospital.\nEventually MS Blake and her friends enlisted the help of a physiotherapist  who happened to be passing on her way to work. She dialed 999 and helped deliver baby Mariah while waiting for emergency services.\nMs Blake said despite the happy ending she was upset she was told to make her own way to the hospital as, being an experienced mum, she knew she did not have the time.\nToday a government spokeswoman said, \"We are disappointed that Ms Blake was not happy with the advice and care she received and will of course investigate any complaint. We are pleased that both Ms Blake and her daughter are well and healthy.\"\n\nQ: Carmen Blake accused the Royal Infirmary of  _  .\n\nA. failing to send an ambulance to help her\nB. having killed her newly-born baby\nC. not taking good care of her and her baby\nD. refusing to admit her into the hospital"}
        #{"role": "user", "content": "Article: This recently-released documentary had some fantastic footage  in it, and a very personal look at many of the astronauts who went to the moon. Overall, that is a very exclusive  club; only about a dozen men ever did it in the history of the world and just eight or nine ever stepped foot on it. Most of them are still alive and they discuss their adventures, insights and personal feelings here.\nOne gets the feeling that the rest of us will never know exactly how beautiful the moon is except to take the astronauts's words about it, because even the pictures on this DVD can't convey that.\nSince this documentary is about 100 minutes long, you get a lot of information. You also get reminded how close two of the three men who went up on that historic first walk on the moon almost didn't get home alive.\nAn absence in this documentary is the most famous astronaut of them all: Neil Armstrong, the first man to step foot on the moon! Apparently, he did not want to be part of this film. One of the astronauts mentions something briefly about Armstrong being somewhat of a \"recluse \" now and it \"being understandable with what he's gone through\". From what I've read, a lot of people have tried to make money off him in shady ways and so now he's withdrawn  from the public spotlight.\nThis film, a legacy to the Apollo program and the brave men who ran it, should be in every schoolroom. It would make history more interesting to students.\n\nQ: What can we know from the passage?\n\nA. One of the astronauts talks about how beautiful the moon is.\nB. Two of the three men who went to the moon lost their lives.\nC. The documentary would make more students interested in history.\nD. The astronauts talk about their adventures, insights and excitement."}
        #{"role": "user", "content": "Article: Sicily, an island of Italy, is home to beautiful beaches, outstanding food, and a bit of Italian history on every corner. It's located just southwest of the Italian mainland and it's the largest Mediterranean island.\nInvaded by many armies over the centuries, it became the site of Roman and Greek colonies. Those cultures remain to this day.\nAlthough they have a rich culture and history, Sicilian people lead a simple life. Living on land with fertile  soil, most of them work in agriculture, fishing and mining, and of course tourism.\nIn Sicily, most stores and businesses are closed from one to four in the afternoon. Street become crowded around five as people start to go out and engage in a variety of activities. They may take a walk to the shops, enjoy a pastry or just meet up with friends.\nFood is one of the great pleasures of Sicilian people. There is an old Sicilian saying:\"With a contented stomach, your heart is forgiving. With an empty stomach you forgive nothing.\" People will go miles out their way to eat fresh seafood. Pasta is the main food Each region has its seasonal pasta dishes, and every family cook their own specialty. Bread is common too. As another Sicilian saying goes, \"A table without bread is like a day without sunshine.\" In Sicily, bread is always freshly baked or bought, and usually twice a day.\n\nQ: The best title of this passage might be  _  .\n\nA. Sicily--an Island of Italy\nB. Sicily--the Site of Roman and Greek Colonies\nC. Italy's most Beautiful Island\nD. SiTALY'S Tasty Island Culture"}
        #{"role": "user", "content": "Article: America used to have a strong college education system for prison inmates (prisoners). It was seen as a way to _ men and women in prison by helping them go straight when they got out.\nThose taxpayer-supported college classes were put to an end in the 1990s. But New York Governor Andrew Cuomo would like to bring them back in the state, setting off a fierce new debate.\nA number of lawmakers in New York have promised to kill Cuomo's proposal  .\nCuomo says reintroducing taxpayer-funded college classes in New York's prisons is a common-sense plan that will reduce the number of inmates who commit new crimes.\n\"You pay $ 60,000 for a prison cell for a year,\" Cuomo responded. \"You put a guy away for 10 years, and that's $600,000. Right now, chances are almost half. Once he's set free, he's going to come right back.\"\nCuomo says helping inmates get a college education would cost about $ 5,000 a year per person. He argues, \"It's a small amount of money if it keeps that inmate from bouncing back into prison.\"\nBut even some members of the governor's own party hate this idea. State Assemblywoman Addie Russell, whose upstate district includes three state prisons, says taxpayers just won't stand for inmates getting a free college education, while middle-class families struggle to pay for their kids' college fees.\n\"That is the vast majority of feedback   that I'm also getting from my constituents  ,\" she says. \"You know, 'Where is the relief for the rest of the population who obey the law ?' \"\n\"I was very disappointed that the policy had been changed,\" says Gerald Gaes, who served as an expert on college programs for the Federal Bureau of Prisons in the 1990s. In 1994, President Clinton stopped federal student aid programs for inmates.\nGaes says research shows that college classes actually save taxpayers' money over time, by reducing the number of inmates who break the law and wind up back in those expensive prison cells.\n\"It is cost-effective,\" he says. \"Designing prisons that way will have a long-term benefit for New York State.\"\n\nQ: Cuomo does the calculations to prove  _  .\n\nA. almost half of prisoners are likely to come back into prison.\nB. college classes for inmates can save taxpayers' money.\nC. the costs of running prisons in the US are on the rise.\nD. it is very difficult to reduce the number of inmates"}


        # mbpp
        #{"role": "user", "content": "Write a function to get the sum of the digits of a non-negative integer.\nYour code should pass these tests:\n\nassert sum_digits(345)==12\nassert sum_digits(12)==3\nassert sum_digits(97)==16"}
        #{"role": "user", "content": "Write a function to extract values between quotation marks from a string.\nYour code should pass these tests:\n\nassert extract_values('\"Python\", \"PHP\", \"Java\"')==['Python', 'PHP', 'Java']\nassert extract_values('\"python\",\"program\",\"language\"')==['python','program','language']\nassert extract_values('\"red\",\"blue\",\"green\",\"yellow\"')==['red','blue','green','yellow']"}
        #{"role": "user", "content": "Write a python function to find the difference between largest and smallest value in a given list.\nYour code should pass these tests:\n\nassert big_diff([1,2,3,4]) == 3\nassert big_diff([4,5,12]) == 8\nassert big_diff([9,2,3]) == 7"}
        #{"role": "user", "content": "Write a function to remove uneven elements in the nested mixed tuple.\nYour code should pass these tests:\n\nassert extract_even((4, 5, (7, 6, (2, 4)), 6, 8)) == (4, (6, (2, 4)), 6, 8)\nassert extract_even((5, 6, (8, 7, (4, 8)), 7, 9)) == (6, (8, (4, 8)))\nassert extract_even((5, 6, (9, 8, (4, 6)), 8, 10)) == (6, (8, (4, 6)), 8, 10)"}
        #{"role": "user", "content": "Write a function to find the dissimilar elements in the given two tuples.\nYour code should pass these tests:\n\nassert find_dissimilar((3, 4, 5, 6), (5, 7, 4, 10)) == (3, 6, 7, 10)\nassert find_dissimilar((1, 2, 3, 4), (7, 2, 3, 9)) == (1, 4, 7, 9)\nassert find_dissimilar((21, 11, 25, 26), (26, 34, 21, 36)) == (34, 36, 11, 25)"}
        #{"role": "user", "content": "Write a function to find sum and average of first n natural numbers.\nYour code should pass these tests:\n\nassert sum_average(10)==(55, 5.5)\nassert sum_average(15)==(120, 8.0)\nassert sum_average(20)==(210, 10.5)"}

        # mmlu
        #{"role": "user", "content": "Which of the following represents an accurate statement concerning arthropods?\n\nA) They possess an exoskeleton composed primarily of peptidoglycan.\nB) They possess an open circulatory system with a dorsal heart.\nC"}
        #{"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.What is the embryological origin of the hyoid bone?\n\nA) The first pharyngeal arch\nB) The first and second pharyngeal arches\nC) The second pharyngeal arch\nD) The second and third pharyngeal arches"}
    ],

    "max_tokens": 8192,  # Sets the max_tokens for all models
    "temperature": 0,  # Sets the tempurature for all models
    "extra_body": {
        "output_path": "outputs/test1.jsonl"
    }
    # "extra_body": {
    #     "agg_model": "Qwen/Qwen2.5-72B-Instruct",
    #     "reference_models": [
    #         "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    #         "Qwen/Qwen2.5-72B-Instruct",
    #         "mistralai/Mistral-Large-Instruct-2411"
    #     ],
    #     "rounds": 2,
    #     "mode": "lc_moa"
    # },
    # "extra_body": {
    #     "agg_model": "claude-3-5-sonnet-20241022",
    #     "reference_models": [
    #         "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    #         "Qwen/Qwen2.5-72B-Instruct",
    #         "gpt-4o-2024-08-06",
    #         "claude-3-5-sonnet-20241022"
    #     ],
    #     "rounds": 2,
    #     "mode": "lc_moa"
    #     # "mode": "standard_moa"
    # }
}
# start = time.time()
response = client.chat.completions.create(
    **payload,
)



#print(f"content: {response.final_answer}")
print(f"content: {response.choices[0].message.content}")
print(f"latency: {response.latency}")
print(f"cost: {response.bill['cost']}")

# end = time.time()
# text = response.choices[0].message.content
# # breakpoint()
# tokens = tokenizer.encode(text, add_special_tokens=False)
# print(response.choices[0].message.content)
# print("total time: {}".format(end-start))
# # print("finish reason: {}".format(response.choices[0].finish_reason))
# print(f"token length: {len(tokens)}")
# # with open('call_moa.txt', 'w') as f:
# #     f.write(response.choices[0].message.content)
# print(response)

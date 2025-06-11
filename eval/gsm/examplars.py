# These examplars are from the Table 20 of CoT paper (https://arxiv.org/pdf/2201.11903.pdf).
EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8"
    }
]

# ये उदाहरण CoT पेपर (https://arxiv.org/pdf/2201.11903.pdf) के तालिका 20 से हैं।
HINDI_EXAMPLARS = [
    {
        "question": "ग्रोव में 15 पेड़ हैं। ग्रोव के कर्मचारी आज ग्रोव में पेड़ लगाएंगे। उनके काम पूरा होने के बाद, 21 पेड़ होंगे। ग्रोव के कर्मचारियों ने आज कितने पेड़ लगाए?",
        "cot_answer": "शुरुआत में 15 पेड़ थे। फिर कुछ और लगाए जाने के बाद 21 पेड़ हो गए। तो 21 - 15 = 6 पेड़ लगाए गए होंगे। तो उत्तर 6 है।",
        "short_answer": "6"
    },
    {
        "question": "यदि पार्किंग स्थल में 3 कारें हैं और 2 और कारें आती हैं, तो पार्किंग स्थल में कितनी कारें हैं?",
        "cot_answer": "शुरुआत में 3 कारें हैं। 2 और कारें आती हैं। 3 + 2 = 5। तो उत्तर 5 है।",
        "short_answer": "5"
    },
    {
        "question": "लेह के पास 32 चॉकलेट थीं और उसकी बहन के पास 42 चॉकलेट थीं। यदि उन्होंने 35 खाईं, तो उनके पास कुल कितनी चॉकलेट बचीं?",
        "cot_answer": "शुरुआत में, लेह के पास 32 चॉकलेट थीं। उसकी बहन के पास 42 चॉकलेट थीं। तो कुल मिलाकर उनके पास 32 + 42 = 74 थीं। 35 खाने के बाद, उनके पास 74 - 35 = 39 बचीं। तो उत्तर 39 है।",
        "short_answer": "39"
    },
    {
        "question": "जेसन के पास 20 लॉलीपॉप थे। उसने डेनी को कुछ लॉलीपॉप दिए। अब जेसन के पास 12 लॉलीपॉप हैं। जेसन ने डेनी को कितने लॉलीपॉप दिए?",
        "cot_answer": "जेसन के पास शुरुआत में 20 लॉलीपॉप थे। डेनी को कुछ देने के बाद उसके पास 12 बचे। तो उसने डेनी को 20 - 12 = 8 दिए। तो उत्तर 8 है।",
        "short_answer": "8"
    },
    {
        "question": "शॉन के पास पाँच खिलौने हैं। क्रिसमस के लिए, उसे अपनी माँ और पिताजी दोनों से दो-दो खिलौने मिले। अब उसके पास कितने खिलौने हैं?",
        "cot_answer": "शॉन के पास शुरुआत में 5 खिलौने थे। यदि उसे अपनी माँ और पिताजी दोनों से 2-2 खिलौने मिले, तो यह 4 और खिलौने हैं। 5 + 4 = 9। तो उत्तर 9 है।",
        "short_answer": "9"
    },
    {
        "question": "सर्वर रूम में नौ कंप्यूटर थे। सोमवार से गुरुवार तक, हर दिन पांच और कंप्यूटर लगाए गए। अब सर्वर रूम में कितने कंप्यूटर हैं?",
        "cot_answer": "शुरुआत में 9 कंप्यूटर थे। 4 दिनों में से प्रत्येक दिन 5 और कंप्यूटर जोड़े गए। तो 5 * 4 = 20 कंप्यूटर जोड़े गए। 9 + 20 = 29। तो उत्तर 29 है।",
        "short_answer": "29"
    },
    {
        "question": "माइकल के पास 58 गोल्फ गेंदें थीं। मंगलवार को, उसने 23 गोल्फ गेंदें खो दीं। बुधवार को, उसने 2 और खो दीं। बुधवार के अंत तक उसके पास कितनी गोल्फ गेंदें थीं?",
        "cot_answer": "माइकल के पास शुरुआत में 58 गोल्फ गेंदें थीं। मंगलवार को 23 खोने के बाद, उसके पास 58 - 23 = 35 बचीं। 2 और खोने के बाद, उसके पास 35 - 2 = 33 गोल्फ गेंदें थीं। तो उत्तर 33 है।",
        "short_answer": "33"
    },
    {
        "question": "ओलिविया के पास $23 हैं। उसने $3 प्रत्येक के हिसाब से पाँच बैगेल खरीदे। उसके पास कितने पैसे बचे हैं?",
        "cot_answer": "ओलिविया के पास 23 डॉलर थे। $3 प्रत्येक के हिसाब से 5 बैगेल $15 होंगे। तो उसके पास 23 - 15 डॉलर बचे हैं। 23 - 15 = 8। तो उत्तर 8 है।",
        "short_answer": "8"
    }
]


HINDI_R_EXAMPLARS = [
    {
        "question": "Grove mein 15 ped hain. Grove ke karmachari aaj grove mein ped lagaenge. Unke kaam poora hone ke baad, 21 ped honge. Grove ke karmachariyon ne aaj kitne ped lagaye?",
        "cot_answer": "Shuruaat mein 15 ped the. Phir kuch aur lagaye jaane ke baad 21 ped ho gaye. Toh 21 - 15 = 6 ped lagaye gaye honge. Toh uttar 6 hai.",
        "short_answer": "6"
    },
    {
        "question": "Yadi parking sthal mein 3 caron hain aur 2 aur caron aati hain, toh parking sthal mein kitni caron hain?",
        "cot_answer": "Shuruaat mein 3 caron hain. 2 aur caron aati hain. 3 + 2 = 5. Toh uttar 5 hai.",
        "short_answer": "5"
    },
    {
        "question": "Leah ke paas 32 chocolates thiin aur uski bahan ke paas 42 chocolates thiin. Yadi unhone 35 khaayin, toh unke paas kul kitni chocolates bachiin?",
        "cot_answer": "Shuruaat mein, Leah ke paas 32 chocolates thiin. Uski bahan ke paas 42 chocolates thiin. Toh kul milakar unke paas 32 + 42 = 74 thiin. 35 khaane ke baad, unke paas 74 - 35 = 39 bachiin. Toh uttar 39 hai.",
        "short_answer": "39"
    },
    {
        "question": "Jason ke paas 20 lollipops the. Usne Denny ko kuch lollipops diye. Ab Jason ke paas 12 lollipops hain. Jason ne Denny ko kitne lollipops diye?",
        "cot_answer": "Jason ke paas shuruaat mein 20 lollipops the. Denny ko kuch dene ke baad uske paas 12 bache. Toh usne Denny ko 20 - 12 = 8 diye. Toh uttar 8 hai.",
        "short_answer": "8"
    },
    {
        "question": "Shawn ke paas paanch khilaune hain. Christmas ke liye, use apni Maa aur Pitaji dono se do-do khilaune mile. Ab uske paas kitne khilaune hain?",
        "cot_answer": "Shawn ke paas shuruaat mein 5 khilaune the. Yadi use apni Maa aur Pitaji dono se 2-2 khilaune mile, toh yeh 4 aur khilaune hain. 5 + 4 = 9. Toh uttar 9 hai.",
        "short_answer": "9"
    },
    {
        "question": "Server room mein nau computer the. Somvar se Guruvar tak, har din paanch aur computer lagaye gaye. Ab server room mein kitne computer hain?",
        "cot_answer": "Shuruaat mein 9 computer the. 4 dinon mein se pratyek din 5 aur computer jode gaye. Toh 5 * 4 = 20 computer jode gaye. 9 + 20 = 29. Toh uttar 29 hai.",
        "short_answer": "29"
    },
    {
        "question": "Michael ke paas 58 golf gendein thiin. Mangalwar ko, usne 23 golf gendein kho diin. Budhwar ko, usne 2 aur kho diin. Budhwar ke ant tak uske paas kitni golf gendein thiin?",
        "cot_answer": "Michael ke paas shuruaat mein 58 golf gendein thiin. Mangalwar ko 23 khone ke baad, uske paas 58 - 23 = 35 bachiin. 2 aur khone ke baad, uske paas 35 - 2 = 33 golf gendein thiin. Toh uttar 33 hai.",
        "short_answer": "33"
    },
    {
        "question": "Olivia ke paas $23 hain. Usne $3 pratyek ke hisaab se paanch bagel khareede. Uske paas kitne paise bache hain?",
        "cot_answer": "Olivia ke paas 23 dollar the. $3 pratyek ke hisaab se 5 bagel $15 honge. Toh uske paas 23 - 15 dollar bache hain. 23 - 15 = 8. Toh uttar 8 hai.",
        "short_answer": "8"
    }
]
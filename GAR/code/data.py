import json
import os
from random import sample
from transformers import LlamaTokenizer
from itertools import chain
from collections import OrderedDict

def remove_duplicate_values(dicts):
    value_count = {}
    for d in dicts:
        for values in d.values():
            for value in values:
                value_count[value] = value_count.get(value, 0) + 1
    
    duplicates = set(key for key, count in value_count.items() if count > 1)
    for d in dicts:
        for key, values in d.items():
            d[key] = [value for value in values if value not in duplicates]

def process_samples(json_data):
    processed_data = {}
    for sample in json_data["samples"]:
        subject_e = sample["subject"]
        object_e = sample["object"]
        if subject_e[0] in ['.'] or object_e[0] in ['.']: continue
        if subject_e[-1] in ['.']: subject_e = subject_e[:-1]
        if object_e[-1] in ['.']: object_e = object_e[:-1]  # e.g. D.C. -> D.C
        if object_e not in processed_data:
            processed_data[object_e] = []
        processed_data[object_e].append(subject_e)
        remove_duplicate_values([processed_data])
    return processed_data

boys = [
    'James', 'David',  'Christopher',  'George',  'Ronald',
    'John', 'Richard',  'Daniel',  'Kenneth',  'Anthony',
    'Robert', 'Charles',  'Paul',  'Steven',  'Kevin',
    'Michael', 'Joseph',  'Mark',  'Edward',  'Jason',
    'William',  'Thomas',  'Donald',  'Brian',  'Jeff',]
girls = [
    'Mary','Jennifer', 'Lisa', 'Sandra', 'Michelle',
    'Patricia','Maria', 'Nancy', 'Donna', 'Laura',
    'Linda','Susan', 'Karen', 'Carol', 'Sarah',
    'Barbara','Margaret', 'Betty', 'Ruth', 'Kimberly',
    'Elizabeth', 'Dorothy', 'Helen', 'Sharon', 'Deborah',]

def all_persons(tokenizer): 
    if not hasattr(all_persons, 'boys'):
        # https://www.verywellfamily.com/top-1000-baby-boy-names-2757618
        # https://www.verywellfamily.com/top-1000-baby-girl-names-2757832
        boys = [l.strip() for l in open('boy_names_1000.txt').readlines()]
        girls = [l.strip() for l in open('girl_names_1000.txt').readlines()]

        wp = '' if isinstance(tokenizer, LlamaTokenizer) else ' '
        girls = [name for name in girls if max(len(tokenizer.tokenize(name)), len(tokenizer.tokenize(wp + name))) == 1]
        boys = [name for name in boys if max(len(tokenizer.tokenize(name)), len(tokenizer.tokenize(wp + name))) == 1]
        boys = sample(boys, len(girls))
        all_persons.boys, all_persons.girls = boys, girls
    return all_persons.boys, all_persons.girls

def genders_of_persons(tokenizer=None):
    genders_of_persons.wh = 'who'
    genders_of_persons.sub_wh = 'the one who'
    genders_of_persons.bos = {'child': ' the'}
    _boys, _girls = all_persons(tokenizer) if tokenizer is not None else (boys, girls)
    return {'the boy': boys, 'the girl': girls}, dict(child='', sibling='a person of the same gender as') 
    # return {'boy': _boys, 'girl': _girls}, dict(child='', sibling='a person of the same gender as')  
    # return {'a male person': boys, 'a female person': girls}, dict(child='', sibling='a person of the same gender as')

def kinds_of_things(): 
    kinds_of_things.name = 'kinds of things'
    kinds_of_things.wh = 'which'
    kinds_of_things.sub_wh = 'the thing which'
    return {
    'animal': ['duck', 'goose', 'dog', 'lion', 'cow', 'donkey', 'horse', 'sheep', 'goat', 'tiger', 'cat', 'pig',
            'monkey', 'rabbit', 'elephant', 'wolf', 'deer', 'fox', 'gorilla', 'squirrel', 'mouse'], # 'chicken', 'bear', 'zebra', 'giraffe', 'kangaroo', 21-5, 15-8
    'fruit': ['apple', 'banana', 'pear', 'grapes', 'cherries', 'orange', 'peach', 'plum', 'lemon', 'mango', 'blackberries',
            'blueberries', 'strawberries', 'durian', 'papaya', 'watermelon', 'pineapple', 'kiwi', 'apricot', 'lime'], # may be food too?
    # 'vegetable': ['spinach', 'broccoli', 'lettuce', 'cabbage', 'tomato'],
    'drink': ['tea', 'coffee', 'beer', 'wine', 'whiskey', 'vodka', 'soda', 'juice', 'cocktail'],  # some as alcohol, 21-5, 15-8
    # 'drink': ['tea', 'coffee', 'beer', 'wine', 'whiskey', 'soda', 'juice',    'vodka', 'cocktail'],  # bad order
    'food': ['hamburger', 'burger', 'bread', 'meat', 'pizza', 'cake', 'steak', 'spaghetti',
            # 'biscuits', 'spaghetti', 'chips', 'peanuts', 'nuts', 'pork', 'beef', 'mutton'
            ],  # last three as meat?~L 21-5?~L 15-8
    'weapon': ['gun', 'handgun', 'shotgun', 'rifle',  'pistol', 'revolver', 'grenade', 'cannon'], #'bomb', 'dagger', 'sword',], # 21-5, 15-8, though latter prefers firearm
    # 'color': ['white', 'black', 'red', 'yellow', 'blue', 'green', 'purple', 'pink', 'gray'],  # 15-8
    'insect': ['mosquito', 'beetle', 'bee'], #'spider', 'ant', 'wasp', 'butterfly'],  # , 'fly'
    # 'flower': ['rose', 'tulip', 'lily', 'daisy', 'sunflower'],
    'vehicle': ['car', 'jeep', 'bus', 'taxi', 'motorcycle'],# 'tractor', 'airplane', 'ship', 'bicycle', 'truck', 'train', 'motorbike', 'helicopter', 'carriage',
                # 'subway', 'van', 'boat'],  # transportation
    # 'furniture': ['sofa', 'couch'], #'desk', 'chair', 'table', 'bed', 'bookshelf'],# 'closet', 'wardrobe'],
    # 'tool': ['hammer', 'spanner', 'awl', 'scissors', 'saw', 'shovel', 'screwdriver', 'wrench', 'drill', 'pliers'], #, 'axe' should be weapon?
    'clothes': ['shirt', 'T-shirt', 'jeans', 'jacket', 'pants', 'trousers', 'shoes', 'sweater', 'jersey', 'underwear', 'costume', 'uniform'],#'dress', 'coat', 'socks', 'hat', 'tie', 'skirt', ],
    # 'clothing': ['shirt', 'T-shirt', 'jeans', 'jacket', 'pants', 'trousers', 'shoes', 'sweater', 'underwear', 'costume', 'uniform',   'jersey'],  # bad order
    # 'appliance': ['microwave', 'fridge', 'washer', 'dryer', 'washing machine'],  #, 'oven'
    # 'fish': [],
    # 'plant': ['tree', 'grass', 'bush', 'weed', 'vine'],
    # 'electronic device': ['laptop', 'iPad', 'phone', 'smartphone'], #'computer', 'television', 'camera', 'printer'],
    # 'electronic device': ['iPad', 'phone', 'smartphone',    'laptop'],  # bad order
    'sport': ['football', 'basketball', 'baseball'],# 'volleyball'],  # 'sport or ball?
    'instrument': ['piano', 'violin', 'guitar'],  # musical instrument
    # 'utensil': ['spoon', 'fork', 'knife', 'plate', 'cup', 'bowl', 'pot'],
    # 'stationery': ['pen', 'pencil', 'paper', 'eraser', 'notebook', 'book', 'ruler', 'ink', 'stapler', 'rubber'],
}, dict(child='a kind of', sibling='the thing of the same kind as')


def countries_of_landmarks():
    countries_of_landmarks.wh = 'which country'
    countries_of_landmarks.name = 'countries of landmarks'
    countries_of_landmarks.sub_wh = 'the country which'
    landmarks = process_samples(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_in_country_clean.json'))))
    return landmarks, dict(child='a place in',sibling='a landmark in the same country as')

def occupations_Of_Persons():
    occupations_Of_Persons.wh = 'who'
    occupations_Of_Persons.name = 'occupations of persons'
    occupations_Of_Persons.sub_wh = 'the person who'
    occupations_Of_Persons.bos = {'child': ' the'}
    return {
        'the actor': ['Leonardo DiCaprio', 'Meryl Streep', 'Tom Hanks', 'Jennifer Lawrence', 'Denzel Washington', 'Julia Roberts', 'Brad Pitt', 'Natalie Portman', 'Johnny Depp', 'Charlize Theron'],
        'the athlete': ['LeBron James', 'Serena Williams', 'Cristiano Ronaldo', 'Usain Bolt', 'Lionel Messi', 'Simone Biles', 'Michael Phelps', 'Roger Federer', 'Tom Brady', 'Alex Morgan'],
        'the musician': ['Beyoncé', 'Taylor Swift', 'Drake', 'Adele', 'Kanye West', 'Rihanna', 'Ed Sheeran', 'Lady Gaga', 'Justin Bieber', 'Eminem'],
        'the scientist': ['Albert Einstein', 'Marie Curie', 'Stephen Hawking', 'Jane Goodall', 'Neil deGrasse Tyson', 'Ada Lovelace', 'Nikola Tesla', 'Carl Sagan', 'Rosie Franklin', 'Richard Dawkins'],
        'the architect': ['Frank Lloyd Wright', 'Zaha Hadid', 'Le Corbusier', 'Ieoh Ming Pei', 'Rem Koolhaas', 'Norman Foster', 'Antoni Gaudí', 'Louis Sullivan', 'Mies van der Rohe', 'Maya Lin'],
        'the author': ['J.K. Rowling', 'Stephen King', 'Agatha Christie', 'George Orwell', 'Toni Morrison', 'Ernest Hemingway', 'Jane Austen', 'Harper Lee', 'J.R.R. Tolkien', 'Mark Twain'],
        'the entrepreneur': ['Elon Musk', 'Jeff Bezos', 'Bill Gates', 'Mark Zuckerberg', 'Oprah Winfrey', 'Steve Jobs', 'Richard Branson', 'Warren Buffett', 'Larry Page', 'Sergey Brin'],
        'the doctor': ['Dr. Anthony Fauci', 'Dr. Sanjay Gupta', 'Dr. Mehmet Oz', 'Dr. Jane Goodall', 'Dr. Michio Kaku', 'Dr. Ben Carson', 'Dr. Neil deGrasse Tyson', 'Dr. Temple Grandin', 'Dr. Gabor Maté', 'Dr. Sylvia Earle'],
        'the lawyer': ['Barack Obama', 'Hillary Clinton', 'Ruth Bader Ginsburg', 'Johnnie Cochran', 'Thurgood Marshall', 'Gloria Allred', 'Clarence Darrow', 'Sonia Sotomayor', 'Alan Dershowitz', 'F. Lee Bailey'],
        'the artist': ['Pablo Picasso', 'Vincent van Gogh', 'Leonardo da Vinci', 'Frida Kahlo', 'Georgia O\'Keeffe', 'Claude Monet', 'Salvador Dalí', 'Andy Warhol', 'Michelangelo', 'Jackson Pollock']
    },dict(child = '',sibling='')

def color_of_fruits():
    color_of_fruits.name = 'color of fruits'
    color_of_fruits.wh = 'which'
    color_of_fruits.sub_wh = 'the fruit which'
    colorOfFruits = process_samples(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_of_fruits.json'))))
    # return colorOfFruits,dict(child='skin color',sibling='')
    return colorOfFruits,dict(child='a kind of fruit with skin color of',sibling='')  

def capabilities_of_things():  
    capabilities_of_things.name = 'capabilities of things'
    capabilities_of_things.wh = 'which is used for'
    capabilities_of_things.sub_wh = 'the thing is used for'
    return { 
    'killing': ['dagger', 'knife', 'gun'],
    'cooking': ['oven', 'pot', 'pan'],
    'writing': ['pen', 'pencil', 'chalk', 'biro'],
    'flying': ['plane', 'glider', 'helicopter'],
    'playing': ['piano', 'violin', 'guitar'],
    'driving': ['car', 'truck', 'jeep'],
    'riding': ['bicycle', 'motorcycle', 'horse'],
    'communicating': ['phone', 'telephone', 'telegraph', 'radio'], # internet, email
    'cleaning': ['broom', 'mop', 'vacuum cleaner'],
    'painting': ['brush', 'palette', 'roller', 'spray'],
    'swiming': ['swimsuit', 'goggles', 'swim fins'],
    'calculating': ['computer', 'calculator', 'abacus'],
},dict(child='', sibling='')

def countries_of_cities(): 
    countries_of_cities.name = 'countries of cities'
    countries_of_cities.wh = 'which country'
    countries_of_cities.sub_wh = 'the country which'
    return {
    'China': ['Beijing', 'Shanghai', 'Guangzhou'],
    'Japan': ['Tokyo', 'Osaka', 'Kyoto'],
    'the United Kingdom': ['London', 'Manchester', 'Birmingham'],  # England
    'the United States': ['Washington, D.C', 'New York', 'Los Angeles'],
    'Canada': ['Ottawa', 'Toronto', 'Vancouver'],
    'Australia': ['Canberra', 'Sydney', 'Brisbane'],
    'France': ['Paris', 'Marseille', 'Lyon'],
    'Italy': ['Rome', 'Milan', 'Florence', 'Venice'],
    'Germany': ['Berlin', 'Hamburg', 'Munich'],
    'Spain': ['Madrid', 'Barcelona', 'Valencia'],
    'Switzerland': ['Bern', 'Zurich', 'Geneva'],
    'Brazil': ['Brasília', 'Sao Paulo', 'Rio de Janeiro'],
    'India': ['New Delhi', 'Mumbai', 'Bangalore'],
    'Thailand': ['Bangkok', 'Chiang Mai', 'Pattaya'],
    'South Korea': ['Seoul', 'Busan', 'Incheon'],
    'Russia': ['Moscow', 'Saint Petersburg', 'Novosibirsk'],  # or St. Petersburg
    # 'Turkey': ['Ankara', 'Istanbul', 'Izmir'],
    # 'Argentina': ['Buenos Aires', 'Cordoba', 'Rosario'],
    # 'Mexico': ['Mexico City', 'Guadalajara', 'Monterrey'],
    # 'Egypt': ['Cairo', 'Alexandria'],
    # 'Portugal': ['Lisbon', 'Porto'],
}, dict(child='a city of', sibling='the city in the same country as')

_person_adjs = [
    # [['fat'], ['thin']],
    # [['hot'], ['cold']],
    # [['big'], ['small']],
    # [['insensitive'], ['sensitive']],
    # [['quiet'], ['loud']],  # noisy
    # [['young'], ['old']],
    # [['conscious'], ['unconscious']],
    # [['asleep'], ['awake']],
    # [['male'], ['female']],
    # [['inside'], ['outside']],
    # [['white'], ['black']],
    [['careful', 'cautious'], ['careless', 'incautious']],
    [['happy', 'glad'], ['sad', 'unhappy']],
    [['rich', 'wealthy'], ['poor', 'impoverished']],
    [['clean', 'splotless'], ['dirty', 'filthy']],  # messy
    # [['tidy'], ['untidy']],
    [['honest', 'candid'], ['dishonest', 'fraudulent']],
    [['brave', 'bold', 'adventurous', 'daring'], ['cowardly', 'timid']],
    [['healthy', 'fine'], ['sick', 'unhealthy']],  # fit, well
    [['friendly', 'affable'], ['unfriendly', 'hostile']],
    [['interesting', 'fascinating'], ['boring', 'uninteresting']],  # amusing
    # expanded by gpt-4
    [['beautiful', 'attractive', 'pretty'], ['ugly', 'unattractive']], # gpt3 wrong
    [['gentle', 'tender'], ['harsh', 'severe']],
    # [['good', 'virtuous'], ['bad', 'evil']],
    # [['popular'], ['unpopular']],
    [['comfortable', 'cozy'], ['uncomfortable', 'awkward']],
    [['responsible', 'dependable'], ['irresponsible', 'negligent']],
    [['rational', 'logical'], ['irrational', 'unreasonable']],
    [['safe', 'secure'], ['dangerous', 'hazardous']],  # harmless, safe is not good according to 16-14
    [['knowledgeable', 'informed'], ['ignorant', 'uninformed']],
    [['active', 'energetic', 'lively'], ['passive', 'inactive', 'lethargic', 'listless']],
    # given by code-davinci-002 and expaneded by gpt-4
    [['reliable', 'trustworthy'], ['unreliable', 'undependable']],
    [['successful', 'prosperous'], ['unsuccessful', 'failing']],
    [['lucky', 'fortunate'], ['unlucky', 'unfortunate']],
    [['generous', 'benevolent'], ['stingy', 'miserly']],
    [['correct', 'right'], ['incorrect', 'wrong']],
    [['diligent', 'hardworking'], ['lazy', 'indolent']],
    [['courteous', 'polite'], ['rude', 'impolite']],  # brutal, harsh
    [['clever', 'smart', 'intelligent'], ['stupid', 'foolish']],
    # by gpt-4
    [['strong', 'powerful'], ['weak', 'feeble']],
    [['fast', 'quick'], ['slow', 'sluggish']],
    [['tall', 'high'], ['short', 'low']],
    [['full', 'filled'], ['empty', 'vacant']],
    [['quiet', 'silent'], ['noisy', 'loud']],
    [['sharp', 'keen'], ['dull', 'blunt']],
    [['kind', 'compassionate'], ['cruel', 'heartless']],
    [['ambitious', 'driven'], ['apathetic', 'unmotivated']],
    [['curious', 'inquiring'], ['indifferent', 'uninterested']],
    [['loyal', 'faithful'], ['disloyal', 'unfaithful']],
    [['modest', 'humble'], ['arrogant', 'boastful']],
    [['sociable', 'outgoing'], ['introverted', 'reserved']],  # shy
    [['thoughtful', 'considerate'], ['thoughtless', 'inconsiderate']],
    [['patient', 'tolerant'], ['impatient', 'intolerant']],
    [['creative', 'innovative'], ['unimaginative', 'conventional']],  # gpt3 wrong
    [['punctual', 'timely'], ['tardy', 'late']],
    [['optimistic', 'positive'], ['pessimistic', 'negative']],
    [['humorous', 'witty', 'funny'], ['serious', 'humorless']],
    [['selfish', 'egotistical'], ['selfless', 'altruistic']],
    [['determined', 'decisive', 'resolute'], ['hesitant', 'indecisive', 'tentative']],  # gpt3 wrong
    # [['light', 'bright'], ['dark', 'dim']], # not for person
    # ['warm', 'hot'], ['cool', 'cold']], # not for person

    # [['sane'], ['mad', 'insane']],
    # [['light'], ['heavy', 'dark']],
    # [['serious'], ['funny']],
    # [['cerebral'], ['emotional']],
    # [['insecure'], ['confident']],
    # [['single'], ['married']],
]
def person_adjs():
    person_adjs.name = 'Descriptions of people'
    # person_adjs.wh = 'which'
    # person_adjs.sub_wh = 'the thing which'
    return _person_adjs, dict()

def join_lists(x, dedup=False):
    l = list(chain.from_iterable(x))
    if dedup: l = list(OrderedDict.fromkeys(l)) # list(set(l)) # to keep order
    return l

def positivities_of_adjs():
    return dict(zip(['positive', 'negtive'], map(join_lists, zip(*_person_adjs))))
adjs = join_lists(positivities_of_adjs().values())
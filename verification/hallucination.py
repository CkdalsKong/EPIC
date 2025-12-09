from transformers import BertForSequenceClassification, BertTokenizer
model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

text='''The vehicle is a Volkswagen Truck and Bus, 17 tons, with a 6.10 TCA MWM International engine. The fuel system is electronically controlled to combine diesel and gas.\nThe bi-fuel is only one of the alternative fuel projects involving MWM International. Another of the company's projects, tests the natural gas + diesel in Acteon 6.12 TCE engine adapted to Otto cycle, and which already presents positive results such as fuel economy and emissions reduction. There are also tests with Biodiesel. In the first semester of 2006, vehicles in this project – a VW 17.210 OD bus equipped with an Acteon electronic engine and two trucks, the VW 8.120 and VW 8.140, exceeded 100,000 kilometers testing.\nReferences.'''
wrong_summary = '''The company MWM International is testing alternative fuel projects, including a bi-fuel system combining diesel and gas, as well as natural gas + diesel and biodiesel. The bi-fuel system is electronically controlled and has shown positive results, such as fuel economy and emissions reduction. Vehicles equipped with this technology have exceeded 100,000 kilometers in testing.'''

input_dict = tokenizer(text, wrong_summary, max_length=512, padding='max_length', truncation='only_first', return_tensors='pt')
logits = model(**input_dict).logits
pred = logits.argmax(dim=1)
model.config.id2label[pred.item()] # prints: INCORRECT
print(model.config.id2label[pred.item()])
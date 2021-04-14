navscript_dhhan_mskim_numbering.py : 여러개의 저장되어있는 문장을 테스트 하는 코드. dataset에 있는 text 파일들을 읽어 진행한다.
navscript_numbering_test.py : 입력하는 한개의 문장을 테스트 하는 코드. ( python navscript_numbering_test.py "원하는 문장")의 형식으로 입력하면 판별된다.
Result_parsing.py : 나온 결과들을 class ID가 맞는것들, script가 맞는것들을 각각 나누어 dataset의 폴더 안에 저장한다. (class_correct_script_correct.txt ...의 파일들)
dataset/count.py : class_correct_script_correct.txt 등의 파일들 중에서 class ID 별로 나누어 각각의 ID당 몇개가 있는지 세어서 list로 변환하여 출력한다.
dataset/data_generator_final.py 데이터를 만드는 코드이다.
#구글 시트 불러오기 
import pandas as pd
#ghseetid = "1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8"
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2"
df = pd.read_csv(gsheet_url)
df.head()

#랜덤하게 2명을 뽑아서 보여주는 코드 
import numpy as np

np.random.seed(20240730)
np.random.choice(df["이름"], 2, replace = False) #False라 중복되지 않게 뽑을 수 있다. 


np.random.seed(20240801)
np.random.choice(np.arange(7)+1, replace = False)
np.random.choice(np.arange(4)+1, replace = False)














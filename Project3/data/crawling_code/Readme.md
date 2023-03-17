# 파일명 규칙

샘플: **crawling\__사이트명_\_sample.py**

전체: **crawling\__사이트명_\_all.py**



# 레시피 json 형식

{

	사이트명:
	
		[
		
			{
				
				link: 페이지 url
			
				title: 레시피 제목
				
				ingredients: [ 재료 리스트 ]
				
				time: 조리시간
				
				serving: 분량
				
				recipe: [ 레시피 ]
				
				calories: 칼로리
				
				carbs: 탄수화물
				
				protein: 단백질
				
				total fat: 지방
				
				image: 대표 이미지 링크(src)
				
			},
			
			{ },
			
			{ },
			
			...
			
		]

}

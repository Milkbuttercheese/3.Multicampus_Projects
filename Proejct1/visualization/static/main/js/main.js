        onload=function(){
            // geolocation 사용 가능
            if ("geolocation" in navigator){
                    navigator.geolocation.getCurrentPosition(function(data){
                        //위경도 저장하기
                        var latitude = data.coords.latitude;
                        var longitude = data.coords.longitude;
                        console.log(latitude)
                        console.log(longitude)
                    });

            } else { // geolocation 사용 불가능
                alert('geolocation 사용 불가능');
		    };
        }
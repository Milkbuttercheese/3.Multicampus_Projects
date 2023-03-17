from django.db import models
from .memvalidators import validemail, validpw, agency_validemail, agency_validpw

class MyMembers(models.Model):

    myemail = models.EmailField(max_length=200, validators=[validemail])
    mypassword = models.CharField(max_length=500, validators=[validpw])
    mynickname = models.CharField(max_length=200, primary_key=True)

    def __str__(self):
        return str({'myemail': self.myemail, 'mypassword': self.mypassword, 'mynick': self.mynickname})


class MentalServiceLocation(models.Model):

    public_or_privates = models.CharField(max_length=20)
    categories = models.CharField(max_length=20)
    agency = models.TextField()
    agency_email = models.EmailField(max_length=200, validators=[agency_validemail], null=True)
    agency_password = models.CharField(max_length=500, validators=[agency_validpw], null=True)
    address = models.CharField(max_length=150)
    latitude = models.DecimalField(max_digits=15, decimal_places=6)
    longitude = models.DecimalField(max_digits=15, decimal_places=6)
    phone_numbers = models.CharField(max_length=30, null=True, blank=True)
    introductions= models.TextField()

    def __str__(self):
        return str({'public_or_privates':self.public_or_privates, 'category':self.categories, 'email':self.agency_email, 'password':self.agency_password,'agency':self.agency, 'address':self.address, 'latitude': self.latitude, 'longtitude':self.longitude, 'phone_number':self.phone_numbers, 'introduction':self.introductions})


class MyBoard(models.Model):

    members_mynickname = models.ForeignKey("MyMembers", related_name='mymembers', on_delete=models.CASCADE, db_column="mymembers_mynickname" )
    mental_address = models.ForeignKey("MentalServiceLocation", related_name='mentalservicelocation', on_delete=models.CASCADE, db_column="mentalservicelocation_address" )
    board_title = models.TextField()
    board_content = models.TextField()

    def __str__(self):
        return str({'mynickname': self.members_mynickname, 'mental_address': self.mental_address})


class MemberBoard(models.Model):
    nickname = models.ForeignKey(MyMembers, on_delete=models.CASCADE)
    board = models.ForeignKey(MyBoard, on_delete=models.CASCADE)
    class Meta:
        db_table = 'member_board'

class MentalBoard(models.Model):
    mental = models.ForeignKey(MentalServiceLocation, on_delete=models.CASCADE)
    board = models.ForeignKey(MyBoard, on_delete=models.CASCADE)
    class Meta:
        db_table = 'mental_board'


class Comment(models.Model):
    post = models.ForeignKey(MyBoard, on_delete=models.CASCADE, null=True)
    reply = models.TextField()

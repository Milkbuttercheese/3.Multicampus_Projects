# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class IngredientCode(models.Model):
    ingredient_code_id = models.IntegerField(primary_key=True)
    preprocessed_ingredient = models.CharField(max_length=50, blank=True, null=True)
    ingredient_code = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ingredient_code'


class IngredientPrice(models.Model):
    ingredient_price_id = models.AutoField(primary_key=True)
    date = models.CharField(max_length=50, blank=True, null=True)
    ingredient_code = models.CharField(max_length=50, blank=True, null=True)
    product = models.CharField(max_length=100, blank=True, null=True)
    price = models.CharField(max_length=10, blank=True, null=True)
    per_price = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ingredient_price'


class PinnedRecipe(models.Model):
    pin_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField(blank=True, null=True)
    recipe = models.ForeignKey('Recipe', models.DO_NOTHING, blank=True, null=True)
    date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'pinned_recipe'


class Rating(models.Model):
    rating_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField(blank=True, null=True)
    recipe = models.ForeignKey('Recipe', models.DO_NOTHING, blank=True, null=True)
    selected_recipe_name = models.CharField(max_length=200, blank=True, null=True)
    stars = models.CharField(max_length=5, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'rating'


class Recipe(models.Model):
    recipe_id = models.AutoField(primary_key=True)
    link = models.CharField(max_length=200, blank=True, null=True)
    title = models.CharField(max_length=200, blank=True, null=True)
    image = models.CharField(max_length=300, blank=True, null=True)
    time = models.CharField(max_length=100, blank=True, null=True)
    serving = models.CharField(max_length=100, blank=True, null=True)
    calories = models.CharField(max_length=20, blank=True, null=True)
    carbs = models.CharField(max_length=20, blank=True, null=True)
    protein = models.CharField(max_length=20, blank=True, null=True)
    total_fat = models.CharField(max_length=20, blank=True, null=True)
    recipe = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    category = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recipe'


class RecipeIngredient(models.Model):
    recipe_ingredient_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200, blank=True, null=True)
    preprocessed_ingredient = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recipe_ingredient'


class UserInfo(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=30, blank=True, null=True)
    user_pw = models.CharField(max_length=30, blank=True, null=True)
    email = models.CharField(max_length=50, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    gender = models.CharField(max_length=3, blank=True, null=True)
    height = models.CharField(max_length=10, blank=True, null=True)
    weight = models.CharField(max_length=5, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user_info'

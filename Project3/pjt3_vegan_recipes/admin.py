from django.contrib import admin
from .models import *

admin.site.register(UserInfo)
admin.site.register(RecipeIngredient)
admin.site.register(Recipe)
admin.site.register(Rating)
admin.site.register(PinnedRecipe)
admin.site.register(IngredientPrice)
admin.site.register(IngredientCode)
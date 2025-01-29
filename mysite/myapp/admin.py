from django.contrib import admin
from .models import Analysis,Disease,AnalysisCT,AnalysisSkin


admin.site.register(Analysis)
admin.site.register(Disease)
admin.site.register(AnalysisCT)
admin.site.register(AnalysisSkin)

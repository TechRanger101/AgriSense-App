from rest_framework import serializers

    
class EndDateSerializer(serializers.Serializer):
    end_date = serializers.DateField()


class IndicesSerializer(serializers.Serializer):
    date = serializers.DateField()
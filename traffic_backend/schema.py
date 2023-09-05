import flow.schema
import graphene

class Query(flow.schema.Query, graphene.ObjectType):
    pass

schema = graphene.Schema(query=Query)
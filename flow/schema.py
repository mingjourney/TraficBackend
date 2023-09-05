import graphene
import json
import pandas as pd
import numpy as np
from django.core.cache import cache
from graphene.types.argument import Argument
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, to_date, year, month, col
from . import predict as flow_predict

spark = SparkSession.builder.getOrCreate()

section_related_OD = json.load(open('section_related_OD.json'))
stations = pd.read_csv('station.csv')
station_names = stations['station_name'].unique()

class FlowInfo(graphene.ObjectType):
    time = graphene.String()
    is_workday = graphene.Int()
    flow = graphene.Int()

class FlowDayTypeInfo(graphene.ObjectType):
    workday = graphene.Float()
    weekend = graphene.Float()

class TimeRange(graphene.InputObjectType):
    start = graphene.String()
    end = graphene.String()

class StationFlowInfo(graphene.ObjectType):
    station = graphene.String()
    flow_in = graphene.List(FlowInfo)
    flow_out = graphene.List(FlowInfo)

class StationSingleFlowInfo(graphene.ObjectType):
    station = graphene.String()
    flow_in = graphene.Int()
    flow_out = graphene.Int()

class AgeStructure(graphene.ObjectType):
    male = graphene.List(graphene.Int)
    female = graphene.List(graphene.Int)

class PeakFlow(graphene.ObjectType):
    morning = graphene.List(StationSingleFlowInfo)
    evening = graphene.List(StationSingleFlowInfo)

class LineToLineFlow(graphene.ObjectType):
    source = graphene.String()
    target = graphene.String()
    flow = graphene.Int()

class LineSectionFlow(graphene.ObjectType):
    source = graphene.String()
    target = graphene.String()
    flow = graphene.Int()

class SectionFlow(graphene.ObjectType):
    up = graphene.List(LineSectionFlow)
    down = graphene.List(LineSectionFlow)

class Query(object):
    single_month_flow = graphene.Field(
        graphene.List(FlowInfo),
        time_ym = graphene.String()
    )
    day_type_flow = graphene.Field(
        FlowDayTypeInfo,
        date_range = Argument(TimeRange)
    )
    single_station_flow = graphene.Field(
        StationFlowInfo,
        station = graphene.String(),
        date_range = Argument(TimeRange)
    )
    age_structure = graphene.Field(AgeStructure)
    peak_flow = graphene.Field(
        PeakFlow,
        date_range = Argument(TimeRange)
    )
    all_stations_flow = graphene.Field(
        graphene.List(StationSingleFlowInfo),
        date_range = Argument(TimeRange)
    )
    line_to_line_flow = graphene.Field(
        graphene.List(LineToLineFlow),
        date_range = Argument(TimeRange)
    )
    section_flow = graphene.Field(
        SectionFlow,
        line = graphene.String(),
        date_range = Argument(TimeRange)
    )
    predict_single_month_flow = graphene.Field(
        graphene.List(FlowInfo),
        timesteps = graphene.Int()
    )
    predict_stations_flow = graphene.Field(graphene.List(StationSingleFlowInfo))
    predict_single_station_flow = graphene.Field(
        StationFlowInfo,
        station = graphene.String()
    )
    predict_section_flow = graphene.Field(
        SectionFlow,
        line = graphene.String()
    )
    predict_peak_flow = graphene.Field(PeakFlow)

    def resolve_single_month_flow(self, info, time_ym):
        key = 'single_month_flow_{}'.format(time_ym)
        v = cache.get(key)
        if (v is None):
            time_y, time_m = time_ym.split('-')
            df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = df.filter((year(df['进站时间']) == time_y) & (month(df['进站时间']) == time_m))
            df = df.groupBy(to_date(df['进站时间']).alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df.rename(columns={'count': 'flow'}, inplace=True)
            df['time'] = df['time'].astype('string')
            cache.set(key, df.to_json(orient='records'), 2678400)
            return df.to_dict(orient='records')
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_day_type_flow(self, info, date_range):
        key = 'day_type_flow_{}-{}'.format(date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']))
            df = df.groupBy(to_date('进站时间').alias('time')).count()
            df_workdays = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/workdays.parquet')
            df = df.join(df_workdays, 'time', how='left')
            df = df.groupBy((col('workday') == 1).alias('is_workday')).mean('count').select(col('is_workday'), col('avg(count)').alias('flow_mean'))
            df = df.toPandas()
            df.set_index('is_workday', inplace=True)
            df = df['flow_mean']
            df.index = df.index.astype('int')
            if not (0 in df.index):
                df.loc[0] = 0
            elif not (1 in df.index):
                df.loc[1] = 0
            result = {
                'workday': df.loc[1],
                'weekend': df.loc[0]
            }
            cache.set(key, json.dumps(result), 2678400)
            return result
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_single_station_flow(self, info, station, date_range):
        key = 'single_station_flow_{}_{}-{}'.format(station, date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']) & (col('进站名称') == station))
            df = df.groupBy(to_date('进站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df.rename(columns={'count': 'flow'}, inplace=True)
            df['time'] = df['time'].astype('string')
            flow_in = df.to_dict(orient='records')
            df = trips
            df = df.filter(to_date('出站时间').between(date_range['start'], date_range['end']) & (col('出站名称') == station))
            df = df.groupBy(to_date('出站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df.rename(columns={'count': 'flow'}, inplace=True)
            df['time'] = df['time'].astype('string')
            flow_out = df.to_dict(orient='records')
            result = {
                'station': station,
                'flow_in': flow_in,
                'flow_out': flow_out
            }
            cache.set(key, json.dumps(result), 2678400)
            return result
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_age_structure(self, info):
        df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/users.parquet')
        df = df.groupBy(['性别', '出生年份']).count().toPandas()
        df['年龄'] = 2021 - df['出生年份']
        age_groups = pd.cut(df['年龄'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        df = df.groupby(['性别', age_groups])['count'].sum().unstack(0)
        return {
            'male': df[0].to_list(),
            'female': df[1].to_list()
        }

    def resolve_peak_flow(self, info, date_range):
        key = 'peak_flow_{}-{}'.format(date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']))
            morning_in = df.filter(hour(df['进站时间']).between(7, 8))
            morning_in = morning_in.groupBy('进站名称').count()
            morning_in = morning_in.toPandas()
            morning_in = pd.DataFrame(zip(morning_in['进站名称'], morning_in['count']), columns=['station', 'flow_in'])
            evening_in = df.filter(hour(df['进站时间']).between(16, 18))
            evening_in = evening_in.groupBy('进站名称').count()
            evening_in = evening_in.toPandas()
            evening_in = pd.DataFrame(zip(evening_in['进站名称'], evening_in['count']), columns=['station', 'flow_in'])
            df = trips
            df = df.filter(to_date('出站时间').between(date_range['start'], date_range['end']))
            morning_out = df.filter(hour(df['出站时间']).between(7, 8))
            morning_out = morning_out.groupBy('出站名称').count()
            morning_out = morning_out.toPandas()
            morning_out = pd.DataFrame(zip(morning_out['出站名称'], morning_out['count']), columns=['station', 'flow_out'])
            evening_out = df.filter(hour(df['出站时间']).between(16, 18))
            evening_out = evening_out.groupBy('出站名称').count()
            evening_out = evening_out.toPandas()
            evening_out = pd.DataFrame(zip(evening_out['出站名称'], evening_out['count']), columns=['station', 'flow_out'])
            morning = pd.merge(morning_in, morning_out, on="station", how="left")
            morning.fillna(0, inplace=True)
            morning = morning.to_dict(orient='records')
            evening = pd.merge(evening_in, evening_out, on="station", how="left")
            evening.fillna(0, inplace=True)
            evening = evening.to_dict(orient='records')
            result = {
                'morning': morning,
                'evening': evening
            }
            cache.set(key, json.dumps(result), 2678400)
            return result
        else:
            cache.expire(key, 2678400)
            return json.loads(v)
        
    def resolve_all_stations_flow(self, info, date_range):
        key = 'all_stations_flow_{}-{}'.format(date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']))
            df = df.groupBy('进站名称').count()
            df = df.toPandas()
            df.rename(columns={'进站名称': 'station', 'count': 'flow_in'}, inplace=True)
            flow_in = df
            df = trips
            df = df.filter(to_date('出站时间').between(date_range['start'], date_range['end']))
            df = df.groupBy('出站名称').count()
            df = df.toPandas()
            df.rename(columns={'出站名称': 'station', 'count': 'flow_out'}, inplace=True)
            flow_out = df
            df = pd.merge(flow_in, flow_out, on="station", how="left")
            df.fillna(0)
            cache.set(key, df.to_json(orient='records'), 2678400)
            return df.to_dict(orient='records')
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_line_to_line_flow(self, info, date_range):
        key = 'line_to_line_flow_{}-{}'.format(date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']))
            df = df.groupby('进站线路', '出站线路').count().toPandas()
            df.rename(columns={'进站线路': 'source', '出站线路': 'target', 'count': 'flow'}, inplace=True)
            cache.set(key, df.to_json(orient='records'), 2678400)
            return df.to_dict(orient='records')
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_section_flow(self, info, line, date_range):
        key = 'section_flow_{}_{}-{}'.format(line, date_range['start'], date_range['end'])
        v = cache.get(key)
        if (v is None):
            df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = df.filter(to_date('进站时间').between(date_range['start'], date_range['end']))
            df = df.groupby('进站名称', '出站名称').count().toPandas()
            df.set_index(['进站名称', '出站名称'], inplace=True)
            section_flow = {
                'up': [],
                'down': []
            }
            line_related_OD = section_related_OD[line]
            for obj in line_related_OD:
                sta_source = obj['source']
                sta_target = obj['target']
                related_OD = obj['related_OD']
                flow_up = 0
                flow_down = 0
                for OD_one in related_OD:
                    if (OD_one['source'], OD_one['target']) in df.index:
                        flow_up += int(df.loc[OD_one['source'], OD_one['target']]['count'])
                    if (OD_one['target'], OD_one['source']) in df.index:
                        flow_down += int(df.loc[OD_one['target'], OD_one['source']]['count'])
                section_flow['up'].append({
                    'source': sta_source,
                    'target': sta_target,
                    'flow': flow_up
                })
                section_flow['down'].append({
                    'source': sta_source,
                    'target': sta_target,
                    'flow': flow_down
                })
            cache.set(key, json.dumps(section_flow), 2678400)
            return section_flow
        else:
            cache.expire(key, 2678400)
            return json.loads(v)

    def resolve_predict_single_month_flow(self, info, timesteps):
        start = pd.to_datetime('2020-07-17').date()
        key = 'predict_single_month_flow_{}'.format(str(start))
        v = cache.get(key)
        if (v is None):
            df = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = df.filter(to_date('进站时间').between(start-timedelta(days=29), start-timedelta(days=1)))
            df = df.groupBy(to_date('进站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df.set_index(pd.to_datetime(df['time']), inplace=True)
            df.rename(columns={'count': 'flow'}, inplace=True)
            df.drop(columns='time', inplace=True)
            df_workdays = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/workdays.parquet')
            df_workdays = df_workdays.filter(col('time').between(start, start+timedelta(days=timesteps-1)))
            df_workdays = df_workdays.toPandas()
            is_workday = df_workdays['workday'] == 1
            is_workday = is_workday.to_list()
            cache.set(key, json.dumps([df.to_dict(orient='records'), is_workday]), 86400)
        else:
            df, is_workday = json.loads(v)
            df = pd.DataFrame(df)
            df.set_index(pd.date_range(start-timedelta(days=29), start-timedelta(days=1)), inplace=True)
        y = flow_predict.predict_n(timesteps, df, is_workday)
        return [{'time': start + timedelta(days=i), 'is_workday': is_workday[i], 'flow': y[i]} for i in range(timesteps)]
    
    def resolve_predict_stations_flow(self, info):
        start = pd.to_datetime('2020-07-17').date()
        key = 'predict_stations_flow_{}'.format(str(start))
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips.filter(to_date('进站时间').between(start-timedelta(days=28), start-timedelta(days=1)))
            df = df.groupBy('进站名称', to_date('进站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df = df.groupby(['进站名称', 'time']).sum().unstack(1).fillna(0)['count']
            df.columns.name = ''
            df.reset_index(inplace=True)
            df.columns = df.columns.astype('string')
            df_in = df
            df = trips.filter(to_date('出站时间').between(start-timedelta(days=28), start-timedelta(days=1)))
            df = df.groupBy('出站名称', to_date('出站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df = df.groupby(['出站名称', 'time']).sum().unstack(1).fillna(0)['count']
            df.columns.name = ''
            df.reset_index(inplace=True)
            df.columns = df.columns.astype('string')
            df_out = df
            cache.set(key, json.dumps([df_in.to_dict(orient='records'), df_out.to_dict(orient='records')]), 86400)
        else:
            df_in, df_out = json.loads(v)
            df_in, df_out = pd.DataFrame(df_in), pd.DataFrame(df_out)
        columns = pd.to_datetime(df_in.columns[1:])
        columns = columns.insert(0, df_in.columns[0])
        df_in.columns = columns
        columns = pd.to_datetime(df_out.columns[1:])
        columns = columns.insert(0, df_out.columns[0])
        df_out.columns = columns
        result_in = flow_predict.predict_station_flow_in(df_in)
        result_out = flow_predict.predict_station_flow_out(df_out)
        result = [{'station': k, 'flow_in': result_in[k], 'flow_out': result_out[k]} for k in result_in]
        return result

    def resolve_predict_single_station_flow(self, info, station):
        start = pd.to_datetime('2020-07-17').date()
        key = 'predict_stations_flow_{}'.format(str(start))
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips.filter(to_date('进站时间').between(start-timedelta(days=28), start-timedelta(days=1)))
            df = df.groupBy('进站名称', to_date('进站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df = df.groupby(['进站名称', 'time']).sum().unstack(1).fillna(0)['count']
            df.columns.name = ''
            df.reset_index(inplace=True)
            df.columns = df.columns.astype('string')
            df_in = df
            df = trips.filter(to_date('出站时间').between(start-timedelta(days=28), start-timedelta(days=1)))
            df = df.groupBy('出站名称', to_date('出站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df = df.groupby(['出站名称', 'time']).sum().unstack(1).fillna(0)['count']
            df.columns.name = ''
            df.reset_index(inplace=True)
            df.columns = df.columns.astype('string')
            df_out = df
            cache.set(key, json.dumps([df_in.to_dict(orient='records'), df_out.to_dict(orient='records')]), 86400)
        else:
            df_in, df_out = json.loads(v)
            df_in, df_out = pd.DataFrame(df_in), pd.DataFrame(df_out)
        columns = pd.to_datetime(df_in.columns[1:])
        columns = columns.insert(0, df_in.columns[0])
        df_in.columns = columns
        df_in = flow_predict.predict_station_flow_in_n(7, df_in)
        df_in = df_in.loc[station].reset_index()
        df_in.columns = ['time', 'flow']
        df_in['time'] = df_in['time'].astype('string')
        columns = pd.to_datetime(df_out.columns[1:])
        columns = columns.insert(0, df_out.columns[0])
        df_out.columns = columns
        df_out = flow_predict.predict_station_flow_out_n(7, df_out)
        df_out = df_out.loc[station].reset_index()
        df_out.columns = ['time', 'flow']
        df_out['time'] = df_out['time'].astype('string')
        result = {
            'station': station,
            'flow_in': df_in.to_dict(orient='records'),
            'flow_out': df_out.to_dict(orient='records')
        }
        return result

    def resolve_predict_section_flow(self, info, line):
        start = pd.to_datetime('2020-07-17').date()
        key = 'predict_section_flow_{}'.format(str(start))
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips.filter(to_date('进站时间').between(start-timedelta(days=28), start-timedelta(days=1)))
            df = df.groupBy('进站名称', '出站名称', to_date('进站时间').alias('time')).count()
            df = df.sort(col('time'))
            df = df.toPandas()
            df.index = df['time']
            df.index = pd.to_datetime(df.index)
            ODMatrix = df.groupby(['进站名称', '出站名称']).resample('D')['count'].sum()
            all_date = ODMatrix.index.levels[2]
            section_flow_sequence_up = []
            section_flow_sequence_down = []
            for date in all_date:
                section_flow_up = []
                section_flow_down = []
                ODMatrix_one = ODMatrix.loc[:, :, date]
                for line in stations['line_name'].unique():
                    line_related_OD = section_related_OD[line]
                    for obj in line_related_OD:
                        related_OD = obj['related_OD']
                        flow_up = 0
                        flow_down = 0
                        for OD_one in related_OD:
                            if (OD_one['source'], OD_one['target']) in ODMatrix_one.index:
                                flow_up += ODMatrix_one.loc[OD_one['source'], OD_one['target']]
                            if (OD_one['target'], OD_one['source']) in ODMatrix_one.index:
                                flow_down += ODMatrix_one.loc[OD_one['target'], OD_one['source']]
                        section_flow_up.append(flow_up)
                        section_flow_down.append(flow_down)
                section_flow_sequence_up.append(section_flow_up)
                section_flow_sequence_down.append(section_flow_down)
            section_flow_sequence_up = np.array(section_flow_sequence_up)
            section_flow_sequence_up = section_flow_sequence_up.T
            section_flow_sequence_down = np.array(section_flow_sequence_down)
            section_flow_sequence_down = section_flow_sequence_down.T
            cache.set(key, json.dumps([
                section_flow_sequence_up.tolist(),
                section_flow_sequence_down.tolist(),
            ]), 86400)
        else:
            section_flow_sequence_up, section_flow_sequence_down = json.loads(v)
            section_flow_sequence_up, section_flow_sequence_down = np.array(section_flow_sequence_up), np.array(section_flow_sequence_down)
        flow_up_vector = flow_predict.predict_section_flow_up(section_flow_sequence_up)
        flow_down_vector = flow_predict.predict_section_flow_down(section_flow_sequence_down)
        section_flow = {
            'up': [],
            'down': []
        }
        index = 0
        for ligne in stations['line_name'].unique():
            line_related_OD = section_related_OD[ligne]
            for obj in line_related_OD:
                if ligne == line:
                    sta_source = obj['source']
                    sta_target = obj['target']
                    section_flow['up'].append({
                        'source': sta_source,
                        'target': sta_target,
                        'flow': flow_up_vector[index]
                    })
                    section_flow['down'].append({
                        'source': sta_source,
                        'target': sta_target,
                        'flow': flow_down_vector[index]
                    })
                index += 1
        return section_flow

    def resolve_predict_peak_flow(self, info):
        start = pd.to_datetime('2020-07-17').date()
        key = 'predict_peak_flow_{}'.format(str(start))
        v = cache.get(key)
        if (v is None):
            trips = spark.read.load('hdfs://localhost:9000/users/gujiaming/traffic/trips.parquet')
            df = trips
            morning_in = df.filter(hour(df['进站时间']).between(7, 8))
            morning_in = morning_in.groupBy('进站名称', to_date('进站时间').alias('time')).count() 
            morning_in = morning_in.sort(col('time'))
            morning_in = morning_in.toPandas()
            morning_in = morning_in.groupby(['进站名称', 'time']).sum().unstack(1).fillna(0)['count']
            for station in station_names:
                if station not in morning_in.index:
                    morning_in.loc[station] = pd.Series([], dtype='float64')
            morning_in = morning_in.fillna(0).sort_values(by='进站名称')
            morning_in.columns.name = ''
            morning_in = morning_in.reset_index()
            morning_in.columns = morning_in.columns.astype('string')
            morning_out = df.filter(hour(df['出站时间']).between(7, 8))
            morning_out = morning_out.groupBy('出站名称', to_date('出站时间').alias('time')).count() 
            morning_out = morning_out.sort(col('time'))
            morning_out = morning_out.toPandas()
            morning_out = morning_out.groupby(['出站名称', 'time']).sum().unstack(1).fillna(0)['count']
            for station in station_names:
                if station not in morning_out.index:
                    morning_out.loc[station] = pd.Series([], dtype='float64')
            morning_out = morning_out.fillna(0).sort_values(by='出站名称')
            morning_out.columns.name = ''
            morning_out = morning_out.reset_index()
            morning_out.columns = morning_out.columns.astype('string')
            evening_in = df.filter(hour(df['进站时间']).between(16, 18))
            evening_in = evening_in.groupBy('进站名称', to_date('进站时间').alias('time')).count() 
            evening_in = evening_in.sort(col('time'))
            evening_in = evening_in.toPandas()
            evening_in = evening_in.groupby(['进站名称', 'time']).sum().unstack(1).fillna(0)['count']
            for station in station_names:
                if station not in evening_in.index:
                    evening_in.loc[station] = pd.Series([], dtype='float64')
            evening_in = evening_in.fillna(0).sort_values(by='进站名称')
            evening_in.columns.name = ''
            evening_in = evening_in.reset_index()
            evening_in.columns = evening_in.columns.astype('string')
            evening_out = df.filter(hour(df['出站时间']).between(16, 18))
            evening_out = evening_out.groupBy('出站名称', to_date('出站时间').alias('time')).count() 
            evening_out = evening_out.sort(col('time'))
            evening_out = evening_out.toPandas()
            evening_out = evening_out.groupby(['出站名称', 'time']).sum().unstack(1).fillna(0)['count']
            for station in station_names:
                if station not in evening_out.index:
                    evening_out.loc[station] = pd.Series([], dtype='float64')
            evening_out = evening_out.fillna(0).sort_values(by='出站名称')
            evening_out.columns.name = ''
            evening_out = evening_out.reset_index()
            evening_out.columns = evening_out.columns.astype('string')
            cache.set(key, json.dumps([
                morning_in.to_dict(orient='records'),
                morning_out.to_dict(orient='records'),
                evening_in.to_dict(orient='records'),
                evening_out.to_dict(orient='records')
            ]), 86400)
        else:
            morning_in, morning_out, evening_in, evening_out = json.loads(v)
            morning_in, morning_out, evening_in, evening_out = pd.DataFrame(morning_in), pd.DataFrame(morning_out), pd.DataFrame(evening_in), pd.DataFrame(evening_out)
        columns = pd.to_datetime(morning_in.columns[1:])
        columns = columns.insert(0, morning_in.columns[0])
        morning_in.columns = columns
        columns = pd.to_datetime(morning_out.columns[1:])
        columns = columns.insert(0, morning_out.columns[0])
        morning_out.columns = columns
        columns = pd.to_datetime(evening_in.columns[1:])
        columns = columns.insert(0, evening_in.columns[0])
        evening_in.columns = columns
        columns = pd.to_datetime(evening_out.columns[1:])
        columns = columns.insert(0, evening_out.columns[0])
        evening_out.columns = columns
        result_morning_in = flow_predict.predict_peak_flow_morning_in(morning_in)
        result_morning_out = flow_predict.predict_peak_flow_morning_out(morning_out)
        result_evening_in = flow_predict.predict_peak_flow_evening_in(evening_in)
        result_evening_out = flow_predict.predict_peak_flow_evening_out(evening_out)
        result = {
            'morning': [{'station': k, 'flow_in': result_morning_in[k], 'flow_out': result_morning_out[k]} for k in result_morning_in],
            'evening': [{'station': k, 'flow_in': result_evening_in[k], 'flow_out': result_evening_out[k]} for k in result_evening_in]
        }
        return result
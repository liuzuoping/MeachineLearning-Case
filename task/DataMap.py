import math
import folium
import pandas as pd
import numpy as np
import webbrowser as wb
import pymssql as pms
import sqlalchemy as sql
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from folium import plugins
from folium.plugins import HeatMap

bl_min = 50000
bl_max = 80000

vt_flag = 1


def get_map(log_user, brand, vt_flag, pc_flag, ml_flag, bl_flag, bl_min, bl_max):
    server = "10.110.3.56"
    user = "sa"
    password = "1qaz2wsx3EDC"
    database = "city_test"
    conn = pms.connect(server, user, password, database)

    log_user = 'xu.chen'

    strSQL = "SELECT * FROM [legend].[compass].[d_user_init_city] where row_num=1 and user_name='" + log_user + "'"
    df_init_city = pd.read_sql(strSQL, conn)

    latitude = df_init_city.loc[0, 'latitude']
    longitude = df_init_city.loc[0, 'longitude']

    region_company_id = str(df_init_city.loc[0, 'region_company_id'])

    # 地图上悬浮显示经纬度
    san_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=12,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='&copy; <a href="http://ditu.amap.com/styles/macaron/">高德地图</a>'
    )

    # vechile track data
    if vt_flag > -1:
        if vt_flag == 2:
            strSQL = "select * from [legend].[compass].[d_vechile_location_city] where region_company_id = " + region_company_id
        else:
            strSQL = "select * from [legend].[compass].[d_vechile_location_city_weekday] where region_company_id = " + region_company_id + " and [weekday]=" + str(
                vt_flag)
        df_gps_range = pd.read_sql(strSQL, conn)

        max_color = max(df_gps_range['location_times'])

        all_boxes = []

        i = 0

        for i in range(len(df_gps_range)):
            upper_left = [df_gps_range.loc[i, 'longitude'] - 0.0075, df_gps_range.loc[i, 'latitude'] + 0.0075]
            upper_right = [df_gps_range.loc[i, 'longitude'] + 0.0075, df_gps_range.loc[i, 'latitude'] + 0.0075]
            lower_right = [df_gps_range.loc[i, 'longitude'] + 0.0075, df_gps_range.loc[i, 'latitude'] - 0.0075]
            lower_left = [df_gps_range.loc[i, 'longitude'] - 0.0075, df_gps_range.loc[i, 'latitude'] - 0.0075]

            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]

            geo_json = {"type": "FeatureCollection",
                        "properties": {
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features": []}

            grid_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                }
            }

            geo_json["features"].append(grid_feature)

            all_boxes.append(geo_json)

            grid = all_boxes

        j = 0

        for j, geo_json in enumerate(grid):
            color = plt.cm.Blues(1 / math.log((max_color + 1) / df_gps_range.loc[j, 'location_times']))
            color = mpl.colors.to_hex(color)

            gj = folium.GeoJson(geo_json,
                                style_function=lambda feature, color=color: {
                                    'fillColor': color,
                                    'color': "white",
                                    'weight': 1,
                                    'line': '5',
                                    'fillOpacity': 0.4,
                                })
            popup = folium.Popup("location heat {}".format(df_gps_range.loc[j, 'location_times']))
            gj.add_child(popup)

            san_map.add_child(gj)

    # block data
    if bl_flag == 1:
        strSQL = "select * from [legend].[compass].[d_block_by_region] where region_company_id = " + region_company_id
        strSQL = strSQL + " and [avg_unit_price] between " + str(bl_min) + " and " + str(bl_max)
        df_gps_range = pd.read_sql(strSQL, conn)

        max_color = max(df_gps_range['avg_unit_price'])

        all_boxes = []

        i = 0

        for i in range(len(df_gps_range)):
            upper_left = [df_gps_range.loc[i, 'longitude'] - 0.0005, df_gps_range.loc[i, 'latitude'] + 0.0005]
            upper_right = [df_gps_range.loc[i, 'longitude'] + 0.0005, df_gps_range.loc[i, 'latitude'] + 0.0005]
            lower_right = [df_gps_range.loc[i, 'longitude'] + 0.0005, df_gps_range.loc[i, 'latitude'] - 0.0005]
            lower_left = [df_gps_range.loc[i, 'longitude'] - 0.0005, df_gps_range.loc[i, 'latitude'] - 0.0005]

            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]

            geo_json = {"type": "FeatureCollection",
                        "properties": {
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features": []}

            grid_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                }
            }

            geo_json["features"].append(grid_feature)

            all_boxes.append(geo_json)

            grid = all_boxes

        j = 0

        for j, geo_json in enumerate(grid):
            color = plt.cm.Greys(1 / math.log((max_color + 1) / df_gps_range.loc[j, 'avg_unit_price']))
            color = mpl.colors.to_hex(color)

            gj = folium.GeoJson(geo_json,
                                style_function=lambda feature, color=color: {
                                    'fillColor': color,
                                    'color': "grey",
                                    'weight': 1,
                                    'line': '2',
                                    'fillOpacity': 0.4,
                                })
            popup = folium.Popup(
                "{}".format(str(df_gps_range.loc[j, 'name']) + " " + str(df_gps_range.loc[j, 'avg_unit_price'])))
            gj.add_child(popup)

            san_map.add_child(gj)

            # power charger data
    if pc_flag == 1:
        strSQL = "SELECT * FROM [legend].[compass].[d_power_cluster] where region_company_id = " + region_company_id
        df_pc = pd.read_sql(strSQL, conn)

        max_fp = df_pc['num'].max()
        color = plt.cm.Greens(0.2)
        color = mpl.colors.to_hex(color)

        df_pc_g = df_pc[df_pc['cluster'] > -1]
        df_pc_i = df_pc[df_pc['cluster'] == -1]

        for lat, lng, label in zip(df_pc_i.latitude, df_pc_i.longitude, str(df_pc_i.num)):
            folium.Circle(
                radius=2,
                location=[lat, lng],
                popup="power charger {}".format(label),
                color=color,
                fill=True,
                fill_color=color,
            ).add_to(san_map)

        for lat, lng, num, rad in zip(df_pc_g.latitude, df_pc_g.longitude, df_pc_g.num, df_pc_g.radius):
            color = plt.cm.Greens(num / max_fp)
            color = mpl.colors.to_hex(color)
            folium.Circle(
                radius=rad,
                location=[lat, lng],
                popup="power charger {}".format(num),
                color=color,
                fill=True,
                fill_color=color,
            ).add_to(san_map)
    # mall data
    if ml_flag == 1:
        strSQL = "SELECT * FROM [legend].[compass].[d_mall_cluster_region] where region_company_id = " + region_company_id
        df_pc = pd.read_sql(strSQL, conn)

        max_fp = df_pc['num'].max()
        color = plt.cm.Reds(0.4)
        color = mpl.colors.to_hex(color)

        df_pc_g = df_pc[df_pc['cluster'] > -1]
        df_pc_i = df_pc[df_pc['cluster'] == -1]

        for lat, lng, label in zip(df_pc_i.latitude, df_pc_i.longitude, str(df_pc_i.num)):
            folium.Circle(
                radius=2,
                location=[lat, lng],
                popup="mall {}".format(label),
                color=color,
                fill=True,
                fill_color=color,
            ).add_to(san_map)

        for lat, lng, num, rad in zip(df_pc_g.latitude, df_pc_g.longitude, df_pc_g.num, df_pc_g.radius):
            color = plt.cm.Reds(num / max_fp)
            color = mpl.colors.to_hex(color)
            folium.Circle(
                radius=rad,
                location=[lat, lng],
                popup="mall {}".format(num),
                color=color,
                fill=True,
                fill_color=color,
            ).add_to(san_map)

    strSQL = "SELECT * FROM [legend].[compass].[d_china_ssss_list_qc] where addr_type in ('门牌号','兴趣点') and brand in ('" + "','".join(
        brand) + "')"
    df_ssss = pd.read_sql(strSQL, conn)

    strSQL = "select * from [legend].[compass].[d_brand_category] where [brand_name] in ('" + "','".join(
        brand) + "') order by [brand_id]"
    df_icon = pd.read_sql(strSQL, conn)

    incidents = folium.map.FeatureGroup()

    # instantiate a mark cluster object for the incidents in the dataframe
    incidents = plugins.MarkerCluster().add_to(san_map)

    # loop through the dataframe and add each data point to the mark cluster
    for bnd in (brand):
        df_brand = df_ssss[df_ssss['brand'] == bnd]
        ico = df_icon[df_icon['brand_name'] == bnd].iloc[0, 0]

        for lat, lng, label, addr in zip(df_brand.latitude, df_brand.longitude, df_brand.store, df_brand.address):
            folium.Marker(
                location=[lat, lng],
                # icon=folium.features.CustomIcon(ico, icon_size=(25,25)),
                icon=folium.features.CustomIcon(
                    'https://image.bitautoimg.com/bt/car/default/images/logo/masterbrand/png/100/m_8_100.png',
                    icon_size=(20, 20)),
                popup=label + "\n\r |" + addr,
            ).add_to(incidents)

    h_file = "_".join(df_icon['brand_id_str'].tolist())
    h_file = df_init_city.loc[0, 'city_name'] + "_" + h_file + ".html"
    h_file = "test.html"
    san_map.save(h_file)
    wb.open(h_file)
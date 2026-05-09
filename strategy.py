#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard periods: 9, 26, 52)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud: future Senkou spans (shifted 26 periods)
    senkou_a_leading = np.roll(senkou_a_6h, 26)
    senkou_b_leading = np.roll(senkou_b_6h, 26)
    senkou_a_leading[:26] = np.nan
    senkou_b_leading[:26] = np.nan
    
    # Get weekly trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_6h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: above 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_leading[i]) or np.isnan(senkou_b_leading[i]) or
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_leading[i], senkou_b_leading[i])
        cloud_bottom = np.minimum(senkou_a_leading[i], senkou_b_leading[i])
        
        if position == 0:
            # Bullish TK cross above cloud in weekly uptrend
            if (tenkan_6h[i] > kijun_6h[i] and  # TK cross bullish
                tenkan_6h[i-1] <= kijun_6h[i-1] and  # crossed just now
                close[i] > cloud_top and  # price above cloud
                close[i] > weekly_ema_6h[i] and  # weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Bearish TK cross below cloud in weekly downtrend
            elif (tenkan_6h[i] < kijun_6h[i] and  # TK cross bearish
                  tenkan_6h[i-1] >= kijun_6h[i-1] and  # crossed just now
                  close[i] < cloud_bottom and  # price below cloud
                  close[i] < weekly_ema_6h[i] and  # weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tenkan_6h[i] < kijun_6h[i] or 
                close[i] < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tenkan_6h[i] > kijun_6h[i] or 
                close[i] > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
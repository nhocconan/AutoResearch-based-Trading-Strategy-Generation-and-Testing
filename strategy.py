#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Volume"
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
    
    # Get 1d data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku to 6h
    tenkan_a = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_a = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_a = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_a = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_a = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 1.8 * 24-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(52, 24, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_a[i]) or np.isnan(kijun_a[i]) or
            np.isnan(senkou_a_a[i]) or np.isnan(senkou_b_a[i]) or
            np.isnan(ema50_1d_a[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan_a[i]
        kijun_val = kijun_a[i]
        senkou_a_val = senkou_a_a[i]
        senkou_b_val = senkou_b_a[i]
        trend = ema50_1d_a[i]
        vol_filter = volume_filter[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Enter long: TK cross bullish, price above cloud, above trend, volume
            if tenkan_val > kijun_val and close[i] > cloud_top and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish, price below cloud, below trend, volume
            elif tenkan_val < kijun_val and close[i] < cloud_bottom and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish or price below cloud
            if tenkan_val < kijun_val or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish or price above cloud
            if tenkan_val > kijun_val or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
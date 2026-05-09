#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Trend_Volume_Simple"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for additional volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h volume filter: current volume > 1.5 * 20-period average
    vol_4h_series = pd.Series(df_4h['volume'].values)
    vol_ma_4h = vol_4h_series.rolling(window=20, min_periods=20).mean().values
    vol_filter_4h = df_4h['volume'].values > (vol_ma_4h * 1.5)
    vol_filter_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_filter_4h)
    
    # 1d volume filter: current volume > 1.3 * 20-period average
    vol_1d_series = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = df_1d['volume'].values > (vol_ma_1d * 1.3)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_filter_4h_aligned[i]) or 
            np.isnan(vol_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_4h_aligned[i]
        vol_ok_4h = vol_filter_4h_aligned[i]
        vol_ok_1d = vol_filter_1d_aligned[i]
        
        if position == 0:
            # Enter long: price above 4h EMA50 with volume confirmation on both timeframes
            if close[i] > trend and vol_ok_4h and vol_ok_1d:
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA50 with volume confirmation on both timeframes
            elif close[i] < trend and vol_ok_4h and vol_ok_1d:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h EMA50
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA50
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
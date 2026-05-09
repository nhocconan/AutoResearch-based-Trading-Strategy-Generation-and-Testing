#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R1 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S1 level
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    camarilla_high_1h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_1h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    vol_avg_1d_1h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_1h[i]) or np.isnan(camarilla_high_1h[i]) or 
            np.isnan(camarilla_low_1h[i]) or np.isnan(vol_avg_1d_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        trend = ema50_4h_1h[i]
        resistance = camarilla_high_1h[i]
        support = camarilla_low_1h[i]
        vol_avg = vol_avg_1d_1h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0 and in_session:
            # Long: break above R1 with volume and above 4h EMA50
            if close[i] > resistance and vol_ok and close[i] > trend:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume and below 4h EMA50
            elif close[i] < support and vol_ok and close[i] < trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 or trend reversal
            if close[i] < support or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above R1 or trend reversal
            if close[i] > resistance or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
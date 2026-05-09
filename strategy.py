#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Camarilla formula: range = high - low
    range_d = high_d - low_d
    # Resistance levels
    r1_d = close_d + range_d * 1.1 / 12
    s1_d = close_d - range_d * 1.1 / 12
    r2_d = close_d + range_d * 1.1 / 6
    s2_d = close_d - range_d * 1.1 / 6
    r3_d = close_d + range_d * 1.1 / 4
    s3_d = close_d - range_d * 1.1 / 4
    
    # Align daily Camarilla levels to 4h timeframe
    r1_d_aligned = align_htf_to_ltf(prices, df_d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_d, s1_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_d, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_d, s2_d)
    r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or
            np.isnan(r2_d_aligned[i]) or
            np.isnan(s2_d_aligned[i]) or
            np.isnan(r3_d_aligned[i]) or
            np.isnan(s3_d_aligned[i]) or
            np.isnan(ema_12h_50_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_d_aligned[i]
        s1_val = s1_d_aligned[i]
        r2_val = r2_d_aligned[i]
        s2_val = s2_d_aligned[i]
        r3_val = r3_d_aligned[i]
        s3_val = s3_d_aligned[i]
        ema_trend = ema_12h_50_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + above 12h EMA50 + volume filter
            if close[i] > r1_val and close[i] > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + below 12h EMA50 + volume filter
            elif close[i] < s1_val and close[i] < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or below 12h EMA50
            if close[i] < s1_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R1 or above 12h EMA50
            if close[i] > r1_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
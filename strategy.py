#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Align to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average for volume confirmation
    vol_series_1d = pd.Series(df_1d['volume'])
    vol_avg_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_level = r1_12h[i]
        s1_level = s1_12h[i]
        ema34 = ema34_12h[i]
        vol_avg = vol_avg_12h[i]
        vol_ok = volume[i] > vol_avg * 2.0  # Volume spike filter
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above daily EMA34
            if close[i] > r1_level and vol_ok and close[i] > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below daily EMA34
            elif close[i] < s1_level and vol_ok and close[i] < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below S1 or trend reversal
            if close[i] < s1_level or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above R1 or trend reversal
            if close[i] > r1_level or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation (using previous day)
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 50:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R1, S1)
    prev_close = df_1d_prev['close'].shift(1).values
    prev_high = df_1d_prev['high'].shift(1).values
    prev_low = df_1d_prev['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12  # R1 = C + 1.1*(H-L)/12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12  # S1 = C - 1.1*(H-L)/12
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 12h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Align all to 12h (primary timeframe)
    r1_12h = align_htf_to_ltf(prices, df_1d_prev, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d_prev, s1)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema50_1d_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_12h[i]
        s1_val = s1_12h[i]
        trend = ema50_1d_12h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
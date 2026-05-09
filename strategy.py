#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Get 1w data for weekly context (trend and volume filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (daily pivot levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (using 1d data)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    r1_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
    s1_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
    
    # Weekly trend filter: 1w EMA34 (longer-term trend)
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 1w volume > 1.5 * 20-week average (institutional interest)
    vol_series_1w = pd.Series(df_1w['volume'].values)
    vol_ma_1w = vol_series_1w.rolling(window=20, min_periods=20).mean().values
    volume_filter_1w = df_1w['volume'].values > (vol_ma_1w * 1.5)
    
    # Align all to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1w_4h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_filter_4h = align_htf_to_ltf(prices, df_1w, volume_filter_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for weekly EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema34_1w_4h[i]) or np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_4h[i]
        s1_val = s1_4h[i]
        trend = ema34_1w_4h[i]
        vol_filter = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above weekly trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below weekly trend
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
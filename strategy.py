#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous week's close, high, low for weekly trend
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Weekly trend: EMA50 of weekly close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Previous day's close, high, low for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 4
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Volume filter: current day volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for weekly EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_trend = ema50_1w_aligned[i]
        vol_filter = volume_aligned[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above weekly trend
            if close[i] > r1_val and close[i] > weekly_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below weekly trend
            elif close[i] < s1_val and close[i] < weekly_trend and vol_filter:
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
# 1d_WeeklyPivot_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily chart strategy using weekly pivot R1/S1 breakouts with 1-week EMA trend filter and volume confirmation. 
# Weekly pivots provide strong structural support/resistance; trend filter ensures alignment with weekly momentum; 
# volume confirmation filters weak breakouts. Designed for 30-100 trades over 4 years to minimize fee drag in both bull and bear markets.

timeframe = "1d"
name = "1d_WeeklyPivot_R1_S1_Breakout_1wTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    w_open = df_1w['open'].values
    
    pivot = (w_high + w_low + w_close) / 3
    r1 = 2 * pivot - w_low
    s1 = 2 * pivot - w_high
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike detection: 1.5x average volume (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > weekly R1 with volume spike and price above weekly EMA34
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < weekly S1 with volume spike and price below weekly EMA34
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch weekly S1 (opposite level) or trend failure (price below weekly EMA34)
            if close[i] < s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch weekly R1 (opposite level) or trend failure (price above weekly EMA34)
            if close[i] > r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
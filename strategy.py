#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily timeframe Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-median). 
Uses discrete sizing (0.30) to balance return and fee drag. Minimum holding period of 2 days reduces churn. 
Works in bull markets via breakout continuation and in bear markets by avoiding counter-trend trades via weekly EMA filter.
Target: 50-100 trades over 4 years (12-25/year) to stay within fee drag limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # need 20 for volume median, 50 for weekly EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1d (based on previous day's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 2.0x 20-period median (robust to outliers)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    bars_since_entry = 0
    
    # Start after warmup (need 20-period for volume median, 50 for weekly EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume spike and 1w uptrend
        long_condition = (close[i] > r1[i]) and volume_spike[i] and (close[i] > ema_50_1w_aligned[i])
        # Short logic: break below S1 with volume spike and 1w downtrend
        short_condition = (close[i] < s1[i]) and volume_spike[i] and (close[i] < ema_50_1w_aligned[i])
        
        # Exit logic: opposite Camarilla level (S1/R1) or trend reversal
        exit_long = (close[i] < s1[i]) or (close[i] < ema_50_1w_aligned[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_50_1w_aligned[i])
        
        # Minimum holding period: 2 bars (2 days)
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
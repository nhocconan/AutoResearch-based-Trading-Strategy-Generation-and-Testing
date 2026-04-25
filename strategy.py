#!/usr/bin/env python3
"""
12h Williams %R Reversal + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions on 12h.
Trading with the daily EMA50 trend filters counter-trend signals.
Volume spike confirms reversal strength.
Works in bull/bear by following daily trend while using %R for entry timing.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll compute on 12h data then align to lower timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:  # Williams %R needs at least 14 periods
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h data
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to lower timeframe (1h)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R (14) + EMA50 + VolMA20
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_level = ema_50_1d_aligned[i]
        williams_r_val = williams_r_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        
        # Exit conditions: opposite extreme or middle zone
        if position != 0:
            if position == 1 and (overbought or williams_r_val > -50):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (oversold or williams_r_val < -50):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Williams %R extreme + trend + volume
        if position == 0:
            long_condition = oversold and (curr_close > ema_50_level) and volume_spike
            short_condition = overbought and (curr_close < ema_50_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
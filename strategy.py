#!/usr/bin/env python3
"""
Hypothesis: 6-hour ADX + weekly Williams %R + volume confirmation. 
Long when ADX > 25 (trending) + Williams %R crosses above -80 (oversold bounce) + volume > 1.5x average.
Short when ADX > 25 + Williams %R crosses below -20 (overbought reversal) + volume > 1.5x average.
Exit when Williams %R crosses back through -50 (mean reversion midpoint) or ADX < 20 (trend weak).
Designed for low trade frequency (~15-30/year) to minimize fee drag in both bull and bear markets.
"""

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
    
    # Load 1-week data for Williams %R - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(whigh).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(wlow).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high.values - wclose) / (highest_high.values - lowest_low.values) * -100
    williams_r = np.where((highest_high.values - lowest_low.values) == 0, -50, williams_r)  # avoid div by zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate ADX (14-period) on 6h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr0 = high[i] - low[i]
        tr1 = abs(high[i] - close[i-1])
        tr2 = abs(low[i] - close[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        williams_r_val = williams_r_aligned[i]
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) + Williams %R crosses above -80 (oversold bounce) + volume confirmation
            if (adx_val > 25 and williams_r_val > -80 and 
                i > 14 and williams_r_aligned[i-1] <= -80 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 + Williams %R crosses below -20 (overbought reversal) + volume confirmation
            elif (adx_val > 25 and williams_r_val < -20 and 
                  i > 14 and williams_r_aligned[i-1] >= -20 and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 or ADX < 20 (trend weak)
                if williams_r_val < -50 or adx_val < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 or ADX < 20 (trend weak)
                if williams_r_val > -50 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_WeeklyWilliamsR_VolumeFilter"
timeframe = "6h"
leverage = 1.0
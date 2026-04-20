#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_TrendFilter
Hypothesis: Trade 4h Donchian(20) breakouts with 1d trend filter and volume confirmation.
Long when price breaks above 20-period high with above-average volume and 1d EMA50 uptrend.
Short when price breaks below 20-period low with above-average volume and 1d EMA50 downtrend.
Donchian channels capture breakout momentum; volume confirms institutional interest; 1d trend filters counter-trend noise.
Works in bull/bear: breakouts with volume and trend alignment have edge. Target: 80-150 total trades over 4 years.
"""

name = "4h_1d_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period volume average for volume confirmation
    vol = prices['volume'].values
    vol_avg = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg[i] = np.mean(vol[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian period
    
    for i in range(start_idx, n):
        # Donchian channel: 20-period high/low
        high_20 = np.max(prices['high'].iloc[i-19:i+1])
        low_20 = np.min(prices['low'].iloc[i-19:i+1])
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        avg_volume = vol_avg[i]
        
        # Volume confirmation: above average volume
        vol_confirmed = not np.isnan(avg_volume) and current_volume > avg_volume
        
        # 1d trend filter
        uptrend = not np.isnan(ema_50_aligned[i]) and current_close > ema_50_aligned[i]
        downtrend = not np.isnan(ema_50_aligned[i]) and current_close < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if current_close > high_20 and vol_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif current_close < low_20 and vol_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low or trend reversal
            if current_close < low_20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high or trend reversal
            if current_close > high_20 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
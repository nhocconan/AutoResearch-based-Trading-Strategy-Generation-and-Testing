#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams %R reversal with 1-day EMA trend filter and volume confirmation.
Only trade reversals when Williams %R reaches extreme oversold/overbought levels (>80 or <20)
and price shows rejection at the level, in the direction of the 1-day EMA34 trend.
Uses 1-day Williams %R for mean reversion signals in ranging markets, filtered by daily trend.
Designed for low trade frequency (12-37/year) by requiring extreme %R levels + trend alignment + volume spike.
Works in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R and EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > EMA34 (uptrend) + volume spike
            if williams_r_aligned[i] < -80 and close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < EMA34 (downtrend) + volume spike
            elif williams_r_aligned[i] > -20 and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 or price < EMA34
                if williams_r_aligned[i] > -50 or close[i] < ema34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R < -50 or price > EMA34
                if williams_r_aligned[i] < -50 or close[i] > ema34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0
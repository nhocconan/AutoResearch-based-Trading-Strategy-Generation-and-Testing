#!/usr/bin/env python3
"""
1d_Elder_Ray_Bull_Bear_Power_Trend
Hypothesis: Use daily Elder Ray indicator (Bull Power = EMA(13) - Low, Bear Power = High - EMA(13)) to capture institutional buying/selling pressure, filtered by 200-day EMA trend and volume confirmation. Go long when Bull Power turns positive with volume spike and price above EMA200, short when Bear Power turns positive with volume spike and price below EMA200. Elder Ray works in trending markets by identifying when bulls or bears gain control, making it effective in both bull (buy the dips) and bear (sell the rallies) regimes.
"""

name = "1d_Elder_Ray_Bull_Bear_Power_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    
    # Align to daily timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bull Power turns positive (bulls gaining control) + volume spike + price above EMA200
            if bull_power_aligned[i-1] <= 0 and bull_power_aligned[i] > 0 and vol_spike and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power turns positive (bears gaining control) + volume spike + price below EMA200
            elif bear_power_aligned[i-1] <= 0 and bear_power_aligned[i] > 0 and vol_spike and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or price breaks below EMA200
            if bull_power_aligned[i] <= 0 or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative or price breaks above EMA200
            if bear_power_aligned[i] <= 0 or close[i] > ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
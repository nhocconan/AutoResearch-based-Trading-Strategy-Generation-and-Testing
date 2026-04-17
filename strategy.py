#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Camarilla pivot breakout + 1d volume confirmation + 6h EMA trend filter.
Long when price breaks above 12h Camarilla R4 with volume > 1.5x 20-period 1d average and price > 6h EMA50.
Short when price breaks below 12h Camarilla S4 with volume > 1.5x 20-period 1d average and price < 6h EMA50.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Combines structure (Camarilla), momentum (volume), and trend (EMA) for robustness in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    # Using typical Camarilla multipliers
    diff = high_12h - low_12h
    r4 = close_12h + 1.1 * diff * 1.1 / 2
    s4 = close_12h - 1.1 * diff * 1.1 / 2
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 6h EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema_50_aligned = align_htf_to_ltf(prices, prices, ema_50)  # self-align for same timeframe
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R4 with volume and above EMA50
            if (close[i] > r4_aligned[i] and 
                volume_confirmed and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S4 with volume and below EMA50
            elif (close[i] < s4_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h Camarilla R3 level
            # Calculate R3 for exit: close + 1.1*(high-low)*1.1/4
            r3 = close_12h + 1.1 * diff * 1.1 / 4
            r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h Camarilla S3 level
            # Calculate S3 for exit: close - 1.1*(high-low)*1.1/4
            s3 = close_12h - 1.1 * diff * 1.1 / 4
            s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hCamarilla_R4S4_Breakout_Volume_EMA50"
timeframe = "6h"
leverage = 1.0
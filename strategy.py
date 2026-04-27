#!/usr/bin/env python3
"""
4h_PriceAction_1dSupportResistance_VolumeBreakout
Hypothesis: Uses 1-day high/low as dynamic support/resistance with volume confirmation and ATR-based risk management. Designed for low trade frequency (~20-30 trades/year) by requiring price to break and hold beyond daily extremes with volume surge, working in both trending and ranging markets.
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
    
    # Calculate 1-day high and low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low for support/resistance
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align 1-day levels to 4h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # ATR for volatility filtering and stop management
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ATR and volume
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        daily_high = daily_high_aligned[i]
        daily_low = daily_low_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above daily high with volume confirmation
            if close[i] > daily_high and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below daily low with volume confirmation
            elif close[i] < daily_low and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below daily low or ATR-based stop
            if close[i] < daily_low or (i > 0 and close[i] < close[i-1] - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above daily high or ATR-based stop
            if close[i] > daily_high or (i > 0 and close[i] > close[i-1] + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_PriceAction_1dSupportResistance_VolumeBreakout"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_Liquidity_Imbalance_Takeout
Hypothesis: Price often revisits recent liquidity pools (equal highs/lows) before continuing the trend.
On 6h timeframe, identify swing highs/lows from 1d timeframe as liquidity zones.
Enter long when price sweeps below a 1d swing low then reverses above it with volume confirmation.
Enter short when price sweeps above a 1d swing high then reverses below it with volume confirmation.
Exit when price reaches opposite liquidity zone or shows reversal signals.
Designed to work in both bull (buy dips) and bear (sell rallies) by fading liquidity sweeps.
Targets 15-25 trades/year for low fee drag.
"""

name = "6h_Liquidity_Imbalance_Takeout"
timeframe = "6h"
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
    
    # Load daily data for swing points (liquidity zones)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Identify swing highs and lows on daily timeframe
    # Swing high: higher high than previous and next bar
    # Swing low: lower low than previous and next bar
    swing_high = np.zeros_like(daily_high, dtype=bool)
    swing_low = np.zeros_like(daily_low, dtype=bool)
    
    for i in range(1, len(daily_high) - 1):
        if daily_high[i] > daily_high[i-1] and daily_high[i] > daily_high[i+1]:
            swing_high[i] = True
        if daily_low[i] < daily_low[i-1] and daily_low[i] < daily_low[i+1]:
            swing_low[i] = True
    
    # Extract swing levels
    swing_high_levels = np.where(swing_high, daily_high, np.nan)
    swing_low_levels = np.where(swing_low, daily_low, np.nan)
    
    # Forward fill to get the most recent swing level
    swing_high_series = pd.Series(swing_high_levels)
    swing_low_series = pd.Series(swing_low_levels)
    recent_swing_high = swing_high_series.ffill().bfill().values
    recent_swing_low = swing_low_series.ffill().bfill().values
    
    # Align daily swing levels to 6h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, recent_swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, recent_swing_low)
    
    # Volume confirmation: 24-period moving average (4 days worth of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        swing_high_val = swing_high_aligned[i]
        swing_low_val = swing_low_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price sweeps below swing low then reverses above it with volume
            if low[i] < swing_low_val and close[i] > swing_low_val and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price sweeps above swing high then reverses below it with volume
            elif high[i] > swing_high_val and close[i] < swing_high_val and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches swing high or shows weakness
            if high[i] >= swing_high_val or (close[i] < swing_low_val and low[i] < swing_low_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches swing low or shows strength
            if low[i] <= swing_low_val or (close[i] > swing_high_val and high[i] > swing_high_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
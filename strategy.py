#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Breakout_VolumeConfirm
Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction filter (price above/below weekly pivot) and volume confirmation (>1.5x 20-period average).
Long when price breaks above 20-period high in weekly uptrend (price > weekly pivot) with volume confirmation.
Short when price breaks below 20-period low in weekly downtrend (price < weekly pivot) with volume confirmation.
Exit via opposite Donchian boundary (10-period) or ATR trailing stop (2.5*ATR from extreme).
Weekly pivot provides higher timeframe structure to filter breakouts, reducing false signals in choppy markets.
Volume confirmation ensures breakouts have conviction. Designed for ~50-120 trades over 4 years (12-30/year) via tight Donchian breakout conditions.
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
    
    # Get 1w data for weekly pivot and trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # need at least 5 periods for pivot calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        pivot = pivot_1w_aligned[i]
        
        if position == 0:
            # Only trade in weekly trending regimes (price relative to weekly pivot)
            if close[i] > pivot:  # weekly uptrend regime
                # Donchian breakout: 20-period high
                highest_20 = np.max(high[i-19:i+1]) if i >= 19 else np.max(high[:i+1])
                long_signal = (close[i] > highest_20) and vol_regime[i]
            else:  # weekly downtrend regime
                # Donchian breakdown: 20-period low
                lowest_20 = np.min(low[i-19:i+1]) if i >= 19 else np.min(low[:i+1])
                short_signal = (close[i] < lowest_20) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below 10-period low (opposite Donchian boundary)
            lowest_10 = np.min(low[i-9:i+1]) if i >= 9 else np.min(low[:i+1])
            if close[i] <= atr_stop or close[i] < lowest_10:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above 10-period high (opposite Donchian boundary)
            highest_10 = np.max(high[i-9:i+1]) if i >= 9 else np.max(high[:i+1])
            if close[i] >= atr_stop or close[i] > highest_10:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
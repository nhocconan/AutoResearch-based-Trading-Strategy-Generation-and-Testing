#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Breakout
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend with volume confirmation.
Long when price breaks above 20-bar Donchian high AND weekly pivot > prior weekly pivot (uptrust) with volume > 1.8x 20-bar average.
Short when price breaks below 20-bar Donchian low AND weekly pivot < prior weekly pivot (downtrend) with volume > 1.8x 20-bar average.
Exit via ATR-based trailing stop (2.0*ATR from extreme) or time-based exit (max 6 bars holding).
Designed for ~12-37 trades/year by requiring strong breakouts, weekly trend alignment, and volume confirmation.
Works in bull/bear markets via weekly pivot trend filter; avoids whipsaws via volume confirmation.
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
    
    # Get weekly data for pivot trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H+L+C)/3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get daily data for ATR calculation (HTF for better stability)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channels (20-period) on primary 6h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume regime: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    bars_held = 0     # bars held in current position
    
    # Start index: need warmup for calculations
    start_idx = max(100, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            bars_held += 1 if position != 0 else 0
            continue
        
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_pivot_prev = weekly_pivot_aligned[i-1] if i > 0 else weekly_pivot_val
        
        # Determine weekly trend direction
        weekly_uptrend = weekly_pivot_val > weekly_pivot_prev
        weekly_downtrend = weekly_pivot_val < weekly_pivot_prev
        
        if position == 0:
            bars_held = 0
            # Only trade in alignment with weekly pivot trend
            if weekly_uptrend:
                # Long: break above Donchian high with volume spike
                long_signal = (high[i] > highest_high[i]) and vol_regime[i]
            elif weekly_downtrend:
                # Short: break below Donchian low with volume spike
                short_signal = (low[i] < lowest_low[i]) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            bars_held += 1
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR time-based exit (max 6 bars)
            atr_stop = long_high - 2.0 * atr_1d_aligned[i]
            time_exit = bars_held >= 6
            if close[i] <= atr_stop or time_exit:
                signals[i] = 0.0
                position = 0
                bars_held = 0
        elif position == -1:
            signals[i] = -0.25
            bars_held += 1
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR time-based exit (max 6 bars)
            atr_stop = short_low + 2.0 * atr_1d_aligned[i]
            time_exit = bars_held >= 6
            if close[i] >= atr_stop or time_exit:
                signals[i] = 0.0
                position = 0
                bars_held = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Breakout"
timeframe = "6h"
leverage = 1.0
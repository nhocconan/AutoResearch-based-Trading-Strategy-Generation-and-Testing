#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day volatility breakout + volume confirmation.
# Uses daily ATR breakout from previous day's close to capture momentum after volatility expansion.
# Long when price breaks above previous close + 0.5*ATR with volume > 1.2x average,
# short when breaks below previous close - 0.5*ATR with volume > 1.2x average.
# Designed for low trade frequency (~15-30/year) to minimize fee decay while capturing volatility breakouts.
# Works in bull/bear markets by trading volatility expansions in either direction.

name = "12h_1d_volatility_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Breakout levels: previous close ± 0.5 * ATR
    upper_break = np.roll(close_1d, 1) + 0.5 * atr_14
    lower_break = np.roll(close_1d, 1) - 0.5 * atr_14
    upper_break[0] = np.nan  # First day has no previous close
    lower_break[0] = np.nan
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(volume_1d, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_20[:19] = np.nan
    
    # Align daily levels to 12h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 1 to ensure we have previous close
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.2 * daily average volume
        vol_filter = volume[i] > 1.2 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks above/below volatility bands with volume
        long_break = high[i] > upper_break_aligned[i] and vol_filter
        short_break = low[i] < lower_break_aligned[i] and vol_filter
        
        # Exit conditions: price returns to previous day's close (mean reversion)
        prev_close = np.roll(close_1d, 1)
        prev_close[0] = np.nan
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
        
        exit_long = low[i] <= prev_close_aligned[i] if not np.isnan(prev_close_aligned[i]) else False
        exit_short = high[i] >= prev_close_aligned[i] if not np.isnan(prev_close_aligned[i]) else False
        
        if long_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_break and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
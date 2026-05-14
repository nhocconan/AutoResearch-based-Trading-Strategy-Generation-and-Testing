#!/usr/bin/env python3
"""
4h_1d_TripleBarrier_Breakout_VolumeTrendFilter_v1
Concept: Triple barrier system using daily high/low/close as dynamic barriers.
- Long when price breaks above previous day's high with volume confirmation (>1.8x avg) and above EMA50
- Short when price breaks below previous day's low with volume confirmation (>1.8x avg) and below EMA50
- Exit when price touches the previous day's close (mean reversion to daily value)
- Uses EMA50 for trend filter to avoid counter-trend trades
- Conservative sizing (0.25) to manage drawdown
- Designed for both bull (breakouts work) and bear (mean reversion to daily close works)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TripleBarrier_Breakout_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for barriers
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily barriers: previous day's high, low, close ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first day's values to NaN (no previous day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Align barriers to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 4h: EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        high_val = prev_high_aligned[i]
        low_val = prev_low_aligned[i]
        close_barrier_val = prev_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(high_val) or np.isnan(low_val) or 
            np.isnan(close_barrier_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above previous day's high with volume confirmation and above EMA50
            breakout_long = close_val > high_val
            vol_confirm = vol_ratio_val > 1.8
            
            if breakout_long and vol_confirm and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below previous day's low with volume confirmation and below EMA50
            elif close_val < low_val and vol_confirm and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below previous day's close
            if close_val <= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above previous day's close
            if close_val >= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
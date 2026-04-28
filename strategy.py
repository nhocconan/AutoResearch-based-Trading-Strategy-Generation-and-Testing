#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (H4/L4) for breakout entries.
# Regime filter: 1d Supertrend to avoid choppy markets and align with trend.
# Volume confirmation: current volume > 1.5 * 20-period average.
# Works in bull (long at H4 breakout in uptrend) and bear (short at L4 breakdown in downtrend).
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "12h_Camarilla_H4L4_Breakout_1dSupertrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and Supertrend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10) for Supertrend
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = 0
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band_1d = (high_1d + low_1d) / 2 + atr_mult * atr_1d
    lower_band_1d = (high_1d + low_1d) / 2 - atr_mult * atr_1d
    
    # Initialize Supertrend
    supertrend_1d = np.full_like(close_1d, np.nan, dtype=float)
    direction_1d = np.full_like(close_1d, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1d)):
        if i == 0:
            supertrend_1d[i] = upper_band_1d[i]
            direction_1d[i] = 1
        else:
            if close_1d[i-1] > supertrend_1d[i-1]:
                supertrend_1d[i] = max(lower_band_1d[i], supertrend_1d[i-1])
                direction_1d[i] = 1
            else:
                supertrend_1d[i] = min(upper_band_1d[i], supertrend_1d[i-1])
                direction_1d[i] = -1
    
    # Align Supertrend direction to 12h
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1/2 * (high - low)
    # L4 = close - 1.1/2 * (high - low)
    camarilla_H4_1d = close_1d + (1.1/2) * (high_1d - low_1d)
    camarilla_L4_1d = close_1d - (1.1/2) * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h (use previous day's levels)
    camarilla_H4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4_1d)
    camarilla_L4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4_1d)
    
    # 12h ATR(14) for breakout confirmation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for volume MA and aligned indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_1d_aligned[i]) or 
            np.isnan(camarilla_H4_1d_aligned[i]) or
            np.isnan(camarilla_L4_1d_aligned[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d Supertrend direction
        uptrend_regime = direction_1d_aligned[i] == 1
        downtrend_regime = direction_1d_aligned[i] == -1
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_H4_1d_aligned[i]
        short_breakout = close[i] < camarilla_L4_1d_aligned[i]
        
        # ATR breakout confirmation: breakout must exceed 0.5 * ATR
        long_atr_confirm = (close[i] - camarilla_H4_1d_aligned[i]) > 0.5 * atr_12h[i]
        short_atr_confirm = (camarilla_L4_1d_aligned[i] - close[i]) > 0.5 * atr_12h[i]
        
        long_entry = uptrend_regime and long_breakout and volume_confirm and long_atr_confirm
        short_entry = downtrend_regime and short_breakout and volume_confirm and short_atr_confirm
        
        # Exit: opposite Camarilla level (H4 for long exit, L4 for short exit)
        long_exit = close[i] < camarilla_L4_1d_aligned[i]
        short_exit = close[i] > camarilla_H4_1d_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
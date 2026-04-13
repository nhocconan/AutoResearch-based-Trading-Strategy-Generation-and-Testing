#!/usr/bin/env python3
"""
Hypothesis: 4h 1-day Bollinger Band squeeze breakout with volume confirmation and volume regime filter.
Uses 1-day Bollinger Bands (20,2) for volatility contraction/expansion, 1-day volume > 1.3x 20-period average for breakout confirmation,
and 1-day volume regime (volume > 0.8x 50-period average) to avoid low-volume false breakouts.
Long when price breaks above upper BB during expansion with volume confirmation.
Short when price breaks below lower BB during expansion with volume confirmation.
Targets 20-40 trades per year to minimize fee drag while capturing meaningful moves.
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
    
    # Get 1d data for Bollinger Bands and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Bollinger Bands (20,2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    
    # Calculate Bollinger Band width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_squeeze = bb_width < 0.8 * bb_width_ma_50  # Bollinger Band squeeze
    
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Calculate 1-day volume filters
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume_1d > 1.3 * vol_ma_20  # Volume spike for breakout confirmation
    vol_regime = volume_1d > 0.8 * vol_ma_50   # Volume regime filter (avoid low volume)
    
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or 
            np.isnan(bb_squeeze_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: BB breakout + volume spike + volume regime (avoid low volume false breakouts)
        breakout_long = close[i] > bb_upper_aligned[i]
        breakout_short = close[i] < bb_lower_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime_ok = vol_regime_aligned[i] > 0.5  # True if adequate volume
        not_squeeze = bb_squeeze_aligned[i] < 0.5  # True if not in squeeze (expansion phase)
        
        long_entry = breakout_long and vol_confirm and vol_regime_ok and not_squeeze
        short_entry = breakout_short and vol_confirm and vol_regime_ok and not_squeeze
        
        # Exit when price returns to Bollinger Band middle (mean reversion)
        exit_long = position == 1 and close[i] < bb_middle_aligned[i]
        exit_short = position == -1 and close[i] > bb_middle_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_bb_squeeze_breakout_volume"
timeframe = "4h"
leverage = 1.0
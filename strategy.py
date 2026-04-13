#!/usr/bin/env python3
"""
1d Weekly Bollinger Band Squeeze Breakout with Volume Confirmation
Hypothesis: In BTC/ETH, low volatility periods (BB squeeze) on weekly chart precede strong moves.
Breakout from Bollinger Bands with volume confirmation captures the start of new trends.
Works in both bull and bear markets by capturing volatility expansion after contraction.
Uses weekly Bollinger Bands (20, 2) for squeeze detection, weekly breakout for direction,
and daily volume spike for confirmation. Target: 20-60 trades over 4 years (5-15/year).
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
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    basis_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    dev_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_1w = basis_1w + (2 * dev_1w)
    lower_1w = basis_1w - (2 * dev_1w)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_1w - lower_1w) / basis_1w
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < (bb_width_ma * 0.5)  # Width < 50% of average = squeeze
    
    # Align weekly data to daily
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition.astype(float))
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma_20 * 2.0)  # Volume > 2x average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Bollinger breakout + volume spike + squeeze condition
        breakout_long = close[i] > upper_1w_aligned[i]
        breakout_short = close[i] < lower_1w_aligned[i]
        vol_confirm = volume_spike_aligned[i] > 0.5
        squeeze_active = squeeze_aligned[i] > 0.5
        
        long_entry = breakout_long and vol_confirm and squeeze_active
        short_entry = breakout_short and vol_confirm and squeeze_active
        
        # Exit when price returns to opposite Bollinger Band (mean reversion)
        exit_long = position == 1 and close[i] < lower_1w_aligned[i]
        exit_short = position == -1 and close[i] > upper_1w_aligned[i]
        
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

name = "1d_weekly_bb_squeeze_breakout"
timeframe = "1d"
leverage = 1.0
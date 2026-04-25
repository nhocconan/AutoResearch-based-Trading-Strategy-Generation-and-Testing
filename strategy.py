#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_1wVolRegime
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter and 1w volume regime filter (volume > 1.5x 4-week MA). Uses discrete sizing 0.25 to limit fee drag. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume regime. Donchian provides structure, EMA50 filters counter-trend breaks, weekly volume avoids low-activity false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate 4-week volume MA on 1w for volume regime detection
    vol_ma_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    volume_regime_1w = volume_1w > (1.5 * vol_ma_1w)
    volume_regime_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_regime_1w)
    
    # Calculate Donchian channels (20-period) from completed 4h bars only
    # Use rolling window on completed bars - no look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 4h bar for Donchian calculation (no look-ahead)
    upper_channel = np.roll(high_roll, 1)
    lower_channel = np.roll(low_roll, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), Donchian (20), volume MA (4)
    start_idx = max(50, 20, 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_regime_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + above 1d EMA50 + high volume regime
            long_setup = (close[i] > upper_channel[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_regime_1w_aligned[i]
            # Short: price breaks below lower Donchian + below 1d EMA50 + high volume regime
            short_setup = (close[i] < lower_channel[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_regime_1w_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below lower Donchian OR below 1d EMA50
            if (close[i] < lower_channel[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above upper Donchian OR above 1d EMA50
            if (close[i] > upper_channel[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_1wVolRegime"
timeframe = "4h"
leverage = 1.0
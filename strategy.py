#!/usr/bin/env python3
"""
4h_1d_volatility_squeeze_breakout
Hypothesis: Combines daily volatility squeeze (Bollinger Band width < 20th percentile) with 4h breakout above/below prior day's high/low. 
Volume confirmation ensures breakout authenticity. Works in both bull and bear markets by capturing volatility expansion after contraction.
Targets 20-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for volatility squeeze and prior day levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Bollinger Band width (20, 2)
    close_1d_series = pd.Series(close_1d)
    sma_20 = close_1d_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_values = bb_width.values
    
    # Calculate 20th percentile of BB width for squeeze condition
    bb_width_percentile_20 = np.nanpercentile(bb_width_values, 20)
    volatility_squeeze = bb_width_values <= bb_width_percentile_20
    
    # Prior day high/low for breakout levels
    prior_day_high = high_1d
    prior_day_low = low_1d
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, volatility_squeeze.astype(float))
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_day_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_day_low)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(prior_high_aligned[i]) or 
            np.isnan(prior_low_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: volatility squeeze + breakout with volume expansion
        long_entry = (squeeze_aligned[i] > 0.5) and (high[i] > prior_high_aligned[i]) and (volume_expansion_aligned[i] > 0.5)
        short_entry = (squeeze_aligned[i] > 0.5) and (low[i] < prior_low_aligned[i]) and (volume_expansion_aligned[i] > 0.5)
        
        # Exit conditions: return to prior day's close (mean reversion after expansion)
        prior_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and close[i] <= prior_close_aligned[i]
        exit_short = position == -1 and close[i] >= prior_close_aligned[i]
        
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

name = "4h_1d_volatility_squeeze_breakout"
timeframe = "4h"
leverage = 1.0
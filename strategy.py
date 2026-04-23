#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d Bollinger Band squeeze and volume spike.
- Williams %R(14): Oversold < -80, Overbought > -20
- Bollinger Band squeeze (1d): BB Width < 20th percentile (low volatility regime)
- Volume spike: Current volume > 2.0x 20-period average
- Long: Williams %R crosses above -80 from below + BB squeeze + volume spike
- Short: Williams %R crosses below -20 from above + BB squeeze + volume spike
- Exit: Williams %R crosses opposite threshold (-20 for long, -80 for short) OR BB squeeze ends
- Uses Williams %R for mean reversion in squeeze regimes, effective in both bull/bear markets
- Target: 50-120 total trades over 4 years (12-30/year) to avoid fee drag
- Discrete position sizing: ±0.25
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Bollinger Bands for squeeze detection
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Bollinger Bands (20, 2)
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized width
    
    # BB squeeze: width < 20th percentile (low volatility regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Align 1d indicators to 4h
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14, 50)  # Need sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(bb_squeeze_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R signals
        wr_cross_above_80 = (williams_r[i] > -80) and (williams_r[i-1] <= -80) if i > 0 else False
        wr_cross_below_20 = (williams_r[i] < -20) and (williams_r[i-1] >= -20) if i > 0 else False
        
        if position == 0:
            # Long: Williams %R crosses above -80 + BB squeeze + volume spike
            if wr_cross_above_80 and bb_squeeze_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + BB squeeze + volume spike
            elif wr_cross_below_20 and bb_squeeze_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR BB squeeze ends
            if wr_cross_below_20 or not bb_squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR BB squeeze ends
            if wr_cross_above_80 or not bb_squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_BBSqueeze_VolumeSpike"
timeframe = "4h"
leverage = 1.0
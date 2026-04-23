#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Regime Filter
- Bollinger Bands (20, 2) on 6h identify low volatility squeezes (BB Width < 20th percentile)
- Breakout occurs when price closes outside BB AND volume > 1.5x 20-period average
- 1d volume regime filter: only trade when 1d volume is above its 50-period median (high volume regime)
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by capturing volatility expansion after contraction
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
    
    # Calculate 6h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2 * dev
    lower_band = basis - 2 * dev
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Calculate BB Width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values * 100
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_1d_median = pd.Series(vol_1d).rolling(window=50, min_periods=50).median().values
    vol_1d_median_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for BB Width percentile and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i]) or np.isnan(vol_1d_median_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: BB squeeze breakout with volume confirmation and 1d volume regime
            bb_squeeze = bb_width_percentile[i] < 20  # Low volatility regime
            breakout_up = close[i] > upper_band[i]
            breakout_down = close[i] < lower_band[i]
            volume_spike = volume[i] > 1.5 * vol_ma[i]
            high_volume_regime = volume[i] > vol_1d_median_aligned[i]  # Current 6h volume > 1d median
            
            if bb_squeeze and breakout_up and volume_spike and high_volume_regime:
                signals[i] = 0.25
                position = 1
            elif bb_squeeze and breakout_down and volume_spike and high_volume_regime:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: BB width expands back above 50th percentile (volatility expansion ending)
            # OR opposite breakout occurs
            exit_signal = False
            
            if position == 1:
                # Exit long when BB width > 50th percentile OR downside breakout
                if (bb_width_percentile[i] > 50 or close[i] < lower_band[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when BB width > 50th percentile OR upside breakout
                if (bb_width_percentile[i] > 50 or close[i] > upper_band[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1dVolumeRegime"
timeframe = "6h"
leverage = 1.0
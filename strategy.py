#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze + Volume Breakout
Hypothesis: Bollinger Band width contraction (squeeze) followed by expansion with volume confirms volatility breakouts.
Works in both bull and bear markets by capturing explosive moves after low volatility periods.
Uses Bollinger Bands (20,2) on 6h with volume confirmation to filter false breakouts.
Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_squeeze_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    bb_mean = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_band = bb_mean + (bb_std_dev * bb_std)
    lower_band = bb_mean - (bb_std_dev * bb_std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band width percentile (50-period) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(bb_period, 50)  # For BB and percentile
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bb_mean[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width in lowest 10% percentile (tight consolidation)
        squeeze_condition = bb_width_percentile[i] <= 10
        
        # Expansion condition: BB width expanding from squeeze
        if i >= 1:
            width_expanding = bb_width[i] > bb_width[i-1]
        else:
            width_expanding = False
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmation = volume[i] > vol_ma[i] * 1.5
        
        # Breakout direction: price outside Bollinger Bands
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to middle band OR volatility contraction returns
            if close[i] <= bb_mean[i] or bb_width_percentile[i] <= 5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to middle band OR volatility contraction returns
            if close[i] >= bb_mean[i] or bb_width_percentile[i] <= 5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: squeeze + expansion + volume + breakout
            if squeeze_condition and width_expanding and volume_confirmation:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Bollinger Band squeeze breakout with daily volume confirmation
# Hypothesis: Bollinger Band squeeze indicates low volatility, often preceding breakouts.
# Volume confirms institutional participation. Works in both bull and bear markets
# by capturing volatility expansion moves in either direction. Target: 25-50 trades/year.
name = "4h_bb_squeeze_1d_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 50-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bb_middle[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle band (mean reversion)
            if close[i] < bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above middle band (mean reversion)
            if close[i] > bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above upper band + BB squeeze + volume confirmation
            if close[i] > bb_upper[i] and bb_squeeze[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below lower band + BB squeeze + volume confirmation
            elif close[i] < bb_lower[i] and bb_squeeze[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
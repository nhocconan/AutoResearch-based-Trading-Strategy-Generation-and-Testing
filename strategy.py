#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bands squeeze breakout with volume confirmation.
# Bollinger Band width contraction (squeeze) precedes explosive moves.
# Enter long/short on breakout from Bollinger Bands with volume confirmation.
# Exit when price reverts to mean (BB middle band) or volatility expands.
# Works in both bull and bear markets by capturing volatility breakouts.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_BB_Squeeze_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20-period, 2 standard deviations)
    bb_period = 20
    bb_std = 2
    
    # Calculate BB middle (SMA), upper, lower
    bb_middle = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # Bollinger Band Width (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Squeeze detection: BB width below its 50-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_m = bb_middle[i]
        bb_u = bb_upper[i]
        bb_l = bb_lower[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        squeeze = squeeze_condition[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Enter long on bullish breakout: price above upper BB with squeeze and volume
            if price > bb_u and squeeze and vol_conf:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish breakout: price below lower BB with squeeze and volume
            elif price < bb_l and squeeze and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle BB or volatility expands (squeeze ends)
            if price < bb_m or not squeeze:  # Mean reversion or volatility expansion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle BB or volatility expands
            if price > bb_m or not squeeze:  # Mean reversion or volatility expansion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1d_volatility_breakout_v1
# Hypothesis: Trade volatility breakouts after low volatility periods on daily timeframe.
# Uses Bollinger Band width to identify low volatility regimes and breaks above/below
# Bollinger Bands for entry, with volume confirmation. Works in both bull and bear
# markets by capturing expansion moves after consolidation. Target: 15-25 trades/year.

name = "1d_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mid = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Bollinger Band width percentile (50-period) to identify low volatility
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Volume ratio (current vs 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # Start from sufficient lookback
    start_idx = 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band or volatility contracts
            if close[i] <= bb_mid[i] or bb_width_percentile[i] < 0.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band or volatility contracts
            if close[i] >= bb_mid[i] or bb_width_percentile[i] < 0.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Low volatility regime (bottom 30% of BB width)
            low_vol = bb_width_percentile[i] <= 0.3
            
            # Long: break above upper band with volume confirmation
            if low_vol and close[i] > bb_upper[i] and vol_ratio[i] > 1.5:
                position = 1
                signals[i] = 0.25
            # Short: break below lower band with volume confirmation
            elif low_vol and close[i] < bb_lower[i] and vol_ratio[i] > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals
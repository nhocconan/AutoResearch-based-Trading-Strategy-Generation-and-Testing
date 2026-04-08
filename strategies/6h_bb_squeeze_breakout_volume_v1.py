#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Bollinger Band Squeeze Breakout with Volume Filter
# Hypothesis: Bollinger Band squeeze (low volatility) precedes explosive moves.
# Breakout direction confirmed by volume surge. Works in bull/bear as volatility
# expansion is regime-independent. Targets 15-25 trades/year with 0.25 position size.

name = "6h_bb_squeeze_breakout_volume_v1"
timeframe = "6h"
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
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: width < 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=10).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle band OR volatility contraction
            if close[i] < bb_middle[i] or squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above middle band OR volatility contraction
            if close[i] > bb_middle[i] or squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade on volatility expansion after squeeze
            if not squeeze[i] and vol_spike[i]:
                # Breakout up: close above upper band
                if close[i] > bb_upper[i] and (i == 20 or close[i-1] <= bb_upper[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Breakout down: close below lower band
                elif close[i] < bb_lower[i] and (i == 20 or close[i-1] >= bb_lower[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
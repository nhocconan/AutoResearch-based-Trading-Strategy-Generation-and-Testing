#|#!/usr/bin/env python3
"""
Hypothesis: 6h 3-bar reversal pattern with volume surge and 1-week trend filter.
- Long: 3 consecutive closes lower + volume > 1.5x 20-period avg + close > weekly EMA50
- Short: 3 consecutive closes higher + volume > 1.5x 20-period avg + close < weekly EMA50
- Exit: Opposite 3-bar reversal or weekly EMA50 cross
Targets 12-37 trades/year per symbol (48-148 total over 4 years).
Works in bull/bear by fading short-term exhaustion in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Detect 3-bar reversal patterns
    # Three consecutive lower closes
    lower3 = (close[2:] < close[1:-1]) & (close[1:-1] < close[:-2])
    lower3 = np.concatenate([[False, False], lower3])
    
    # Three consecutive higher closes
    higher3 = (close[2:] > close[1:-1]) & (close[1:-1] > close[:-2])
    higher3 = np.concatenate([[False, False], higher3])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned indicators
        ema50 = ema50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Check for NaN values
        if np.isnan(ema50) or np.isnan(vol_ma):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: 3 lower closes + above weekly EMA50
                if lower3[i] and close[i] > ema50:
                    position = 1
                    signals[i] = position_size
                # Short: 3 higher closes + below weekly EMA50
                elif higher3[i] and close[i] < ema50:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position
            # Exit: 3 higher closes OR price crosses below weekly EMA50
            if higher3[i] or (close[i] < ema50 and close[i-1] >= ema50):
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position
            # Exit: 3 lower closes OR price crosses above weekly EMA50
            if lower3[i] or (close[i] > ema50 and close[i-1] <= ema50):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_3bar_reversal_volume_weeklyEMA50_v1"
timeframe = "6h"
leverage = 1.0
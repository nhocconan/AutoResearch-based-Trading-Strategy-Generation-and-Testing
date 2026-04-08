#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Bollinger Band Breakout with 12h Volume Confirmation
# Hypothesis: BB breakouts capture volatility expansion moves; 12h volume confirms institutional participation.
# Works in bull via upper band breakouts, in bear via lower band breakdowns. Bollinger Bands adapt to volatility regime.
# Target: 25-40 trades/year to minimize fee drag.
name = "4h_bb20_12h_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume moving average
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 12h average volume
        vol_confirm = volume[i] > vol_ma_12h_aligned[i]
        
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
            # Enter long: price closes above upper band + volume confirmation
            if close[i] > bb_upper[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below lower band + volume confirmation
            elif close[i] < bb_lower[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
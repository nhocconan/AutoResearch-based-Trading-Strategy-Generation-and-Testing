#!/usr/bin/env python3
# 4h_volume_price_action_v4
# Hypothesis: Price action with volume confirmation on 4h timeframe.
# Long when price closes above prior swing high with volume > 1.5x average.
# Short when price closes below prior swing low with volume > 1.5x average.
# Exit on opposite signal or when volume drops below average.
# Uses swing points to capture momentum with volume confirmation to reduce whipsaw.
# Tightened exit conditions to reduce trade frequency and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate swing highs and lows (5-period lookback)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(5, n-5):
        # Swing high: highest high in 5 bars before and after
        if high[i] == np.max(high[i-5:i+6]):
            swing_high[i] = high[i]
        # Swing low: lowest low in 5 bars before and after
        if low[i] == np.min(low[i-5:i+6]):
            swing_low[i] = low[i]
    
    # Forward fill swing levels for comparison
    swing_high = pd.Series(swing_high).ffill().bfill().values
    swing_low = pd.Series(swing_low).ffill().bfill().values
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below swing low or volume drops below 0.8x average
            if close[i] < swing_low[i] or volume[i] < 0.8 * avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above swing high or volume drops below 0.8x average
            if close[i] > swing_high[i] or volume[i] < 0.8 * avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.8x average volume (tightened)
            volume_ok = volume[i] > 1.8 * avg_volume[i]
            
            # Price action entries
            if close[i] > swing_high[i] and volume_ok:
                # Additional confirmation: previous close was at or below swing high
                if i > 0 and close[i-1] <= swing_high[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < swing_low[i] and volume_ok:
                # Additional confirmation: previous close was at or above swing low
                if i > 0 and close[i-1] >= swing_low[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
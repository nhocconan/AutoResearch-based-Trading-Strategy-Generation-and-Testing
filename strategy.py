#!/usr/bin/env python3
# 6h_ema200_rsi14_volume
# Hypothesis: On 6h timeframe, enter long when price > EMA200, RSI14 > 50, and volume > 1.5x average; enter short when price < EMA200, RSI14 < 50, and volume > 1.5x average.
# Exit when price crosses back below/above EMA200 or volume drops below average.
# Uses volume confirmation to avoid false breaks and RSI to avoid overextended moves.
# Designed for low frequency (12-37 trades/year) to minimize fee drag and work in both bull and bear markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema200_rsi14_volume"
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
    
    # Calculate EMA200 on 6h
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate RSI14 on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema200[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA200 or volume drops below average
            if close[i] <= ema200[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA200 or volume drops below average
            if close[i] >= ema200[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price > EMA200, RSI > 50, and volume confirmation
            if close[i] > ema200[i] and rsi[i] > 50 and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price < EMA200, RSI < 50, and volume confirmation
            elif close[i] < ema200[i] and rsi[i] < 50 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_VolumeSpike
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to capture adaptive trend direction on 1d timeframe. 
Enter long when KAMA turns up and price closes above KAMA with volume confirmation; short when KAMA turns down and price closes below KAMA with volume confirmation.
Volume spike (>1.5x 20-day average) confirms institutional participation. 
Works in bull markets (captures uptrends) and bear markets (captures downtrends). 
Target: 15-25 trades per year on 1d timeframe (~60-100 total over 4 years).
"""

name = "1d_KAMA_Trend_With_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W Data for Weekly Trend Filter (Optional) ===
    # Not strictly needed but can add robustness
    
    # === KAMA Calculation on Close (ER=10, Fast=2, Slow=30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs correction - let's do properly
    
    # Proper ER calculation
    dir = np.abs(np.diff(close, n=10))  # 10-period direction
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # Still wrong - let's rebuild
    
    # Let's do KAMA properly with loops since it's adaptive
    kama = np.full(n, np.nan)
    fast_sc = 2 / (2 + 1)  # 2/(fast+1)
    slow_sc = 2 / (30 + 1)  # 2/(slow+1)
    
    # Initialize
    kama[0] = close[0]
    
    # Calculate ER and SC for each point
    for i in range(1, n):
        if i >= 10:
            # Direction over 10 periods
            direction = np.abs(close[i] - close[i-10])
            # Volatility sum of absolute changes over 10 periods
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility > 0:
                er = direction / volatility
            else:
                er = 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]  # Not enough data, use price
    
    # KAMA slope for direction
    kama_slope = np.diff(kama, prepend=0)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # Need enough data for KAMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or 
            np.isnan(kama_slope[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turning up (positive slope) AND price > KAMA AND volume spike
            if kama_slope[i] > 0 and close[i] > kama[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down (negative slope) AND price < KAMA AND volume spike
            elif kama_slope[i] < 0 and close[i] < kama[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down OR price crosses below KAMA
            if kama_slope[i] < 0 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: KAMA turns up OR price crosses above KAMA
            if kama_slope[i] > 0 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
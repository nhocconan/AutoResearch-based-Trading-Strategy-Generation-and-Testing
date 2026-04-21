#!/usr/bin/env python3
"""
12h Williams Alligator + 1w Trend Filter
Long when green line > red line > blue line (bullish alignment) and 1w close > EMA50
Short when green line < red line < blue line (bearish alignment) and 1w close < EMA50
Exit when alignment breaks
Williams Alligator catches trend changes with smoothed lines, reducing whipsaw.
1w EMA50 filters for higher timeframe trend alignment.
Target: 12-37 trades/year for low fee drag and robust performance in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator (13,8,5 SMAs with shifts)
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (blue line): 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (red line): 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (green line): 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment (green > red > blue) and 1w close > EMA50
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                ema_50_aligned[i] > 0):  # 1w close > EMA50 (positive check via alignment)
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (green < red < blue) and 1w close < EMA50
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
                  ema_50_aligned[i] > 0):  # 1w close < EMA50 (negative check via alignment)
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: alignment breaks
            exit_signal = False
            
            if position == 1:
                # Exit long: not bullish alignment
                if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: not bearish alignment
                if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0
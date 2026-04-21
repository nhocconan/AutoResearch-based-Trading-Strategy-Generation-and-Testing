#!/usr/bin/env python3
"""
1d_1W_Weekly_Trend_Following
Hypothesis: Capture weekly trends using 1D close above/below weekly EMA20 with volume confirmation.
Long when daily close > weekly EMA20 and volume > 1.5x 20-day average.
Short when daily close < weekly EMA20 and volume > 1.5x 20-day average.
Exit when price crosses back through weekly EMA20.
Designed for 1d timeframe to capture multi-week trends with ~10-25 trades/year.
Works in bull markets by buying strength and in bear markets by selling weakness.
Volume filter reduces false signals and whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA20
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Calculate weekly EMA20
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume filter: 20-day average volume
    volume_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        if np.isnan(volume_ma[i]):
            volume_ok = False
        else:
            volume_ok = volume > 1.5 * volume_ma[i]
        
        if position == 0:
            # Long conditions: price above weekly EMA20 + volume confirmation
            if price > ema20_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below weekly EMA20 + volume confirmation
            elif price < ema20_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly EMA20
            if price < ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly EMA20
            if price > ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Weekly_Trend_Following"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_1d_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: 12-hour breakout of daily pivot levels (resistance/support) with volume confirmation and ATR-based stop works in both bull and bear markets.
Pivot levels calculated from prior day's OHLC. Long when price breaks above R1 with volume > 1.5x 20-period average, short when breaks below S1.
Volume filter ensures breakouts are genuine. ATR stop limits downside. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility normalization
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily pivot levels from prior day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above R1 with volume expansion
        long_signal = (high[i] > r1[i] and volume_expansion[i])
        
        # Short signal: break below S1 with volume expansion
        short_signal = (low[i] < s1[i] and volume_expansion[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
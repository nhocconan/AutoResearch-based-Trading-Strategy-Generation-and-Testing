#!/usr/bin/env python3
"""
4h_12h_Multiplier_Volume_Filtered_Breakout_V1
Hypothesis: Use 12h average true range multiplier for dynamic breakout thresholds with volume confirmation.
Long when price breaks above 12h close + k*ATR(12h) with volume > 1.5x average.
Short when price breaks below 12h close - k*ATR(12h) with volume > 1.5x average.
Uses 12h timeframe for structure, 4h for entry timing to reduce whipsaw.
Dynamic threshold adapts to volatility, reducing trades in choppy markets.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
Works in bull/bear via volatility-adaptive breakout levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for dynamic breakout levels
    df_12h = get_htf_data(prices, '12h')
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h ATR(10) for volatility measurement
    tr1 = np.maximum(high_12h - low_12h, np.absolute(high_12h - np.roll(close_12h, 1)))
    tr2 = np.absolute(np.roll(close_12h, 1) - low_12h)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Dynamic multiplier: 0.5 in low vol, 1.0 in high vol (inverse relationship)
    # Normalize ATR to 0-1 range over 50 periods
    atr_ma = pd.Series(atr_10).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr_10 / atr_ma, 1.0)
    # Invert ratio: low ATR ratio = high volatility = smaller multiplier
    multiplier = np.where(atr_ratio <= 1.0, 0.5 + 0.5 * (2.0 - atr_ratio), 0.5)
    multiplier = np.clip(multiplier, 0.3, 1.0)  # bound between 0.3 and 1.0
    
    # Calculate dynamic breakout levels
    upper_break = close_12h + multiplier * atr_10
    lower_break = close_12h - multiplier * atr_10
    
    # Align all 12h data to 4h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_12h, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_12h, lower_break)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ATR and MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper level with volume confirmation
            if close[i] > upper_break_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower level with volume confirmation
            elif close[i] < lower_break_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below 12h close (mean reversion) or volume fails
            if close[i] < close_12h[i] or not vol_confirm[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 12h close or volume fails
            if close[i] > close_12h[i] or not vol_confirm[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Multiplier_Volume_Filtered_Breakout_V1"
timeframe = "4h"
leverage = 1.0
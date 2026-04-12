#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout_volatility_regime
Hypothesis: Combine 12h Camarilla breakout with volatility regime filter (ATR ratio) and volume confirmation.
In bull markets: trend-following breakouts work. In bear markets: volatility filter reduces false signals during low-volatility chop.
Uses 4h timeframe with 12h HTF for structure. Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""

name = "4h_12h_camarilla_breakout_volatility_regime"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla and ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's range
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # Camarilla levels (based on previous 12h bar)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # ATR for volatility filter (14-period ATR on 12h)
    tr1 = np.abs(np.subtract(high_12h, low_12h))
    tr2 = np.abs(np.subtract(high_12h, np.roll(close_12h, 1)))
    tr3 = np.abs(np.subtract(low_12h, np.roll(close_12h, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_12h / atr_ma  # >1 = high volatility, <1 = low volatility
    
    # Align Camarilla levels and ATR ratio to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when volatility is elevated (ATR ratio > 0.8)
        vol_regime = atr_ratio_aligned[i] > 0.8
        
        # Long entry: close breaks above R4 with volume and volatility regime
        if (close[i] > r4_aligned[i] and vol_confirm[i] and vol_regime and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and volatility regime
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and vol_regime and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
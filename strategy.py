#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d Trend Filter and Volume Confirmation
Long: Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA(50) AND volume > 1.5x 6s volume SMA(20)
Short: Bull Power < 0 AND Bear Power > 0 AND close < 1d EMA(50) AND volume > 1.5x 6s volume SMA(20)
Exit: Bull Power and Bear Power same sign (both positive or both negative)
Uses 1d EMA for trend, Elder Ray for momentum/balance, volume for confirmation
Target: 15-30 trades/year per symbol (60-120 total over 4 years)
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray (standard period)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High minus EMA
    bear_power = low - ema_13   # Low minus EMA
    
    # Calculate 6s volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 20, 13)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        ema_1d_val = ema_50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price above 1d EMA + volume > 1.5x SMA
            if bull > 0 and bear < 0 and price > ema_1d_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND price below 1d EMA + volume > 1.5x SMA
            elif bull < 0 and bear > 0 and price < ema_1d_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power and Bear Power same sign (both >= 0 or both <= 0)
            if (bull >= 0 and bear >= 0) or (bull <= 0 and bear <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power and Bear Power same sign (both >= 0 or both <= 0)
            if (bull >= 0 and bear >= 0) or (bull <= 0 and bear <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0
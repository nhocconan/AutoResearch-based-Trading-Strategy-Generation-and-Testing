#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d EMA Trend + Volume Spike
Long: Bull Power > 0, price > 1d EMA50, volume > 2x 6m volume SMA(20)
Short: Bear Power < 0, price < 1d EMA50, volume > 2x 6m volume SMA(20)
Exit: Opposite signal or price crosses 1d EMA50
Uses Elder Ray (Bull/Bear Power) to measure trend strength behind price moves.
Designed to work in both bull and bear markets by filtering with 1d trend.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6m volume SMA(20) for volume filter
    vol_sma_6m = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_6m[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_6m[i]
        ema_50_val = ema_50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) + price > 1d EMA50 + volume spike
            if bull > 0 and price > ema_50_val and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) + price < 1d EMA50 + volume spike
            elif bear < 0 and price < ema_50_val and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power > 0 (selling pressure appears) or price < 1d EMA50
            if bear > 0 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power < 0 (buying pressure appears) or price > 1d EMA50
            if bull < 0 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Enters long when Bull Power > 0, volume > 1.5x daily average, and 1-day close > EMA34 (uptrend).
Enters short when Bear Power < 0, volume > 1.5x daily average, and 1-day close < EMA34 (downtrend).
Uses EMA13 for power calculation and EMA34 for trend filter to reduce noise.
Targets 15-30 trades/year per symbol to minimize fee drift while capturing institutional sentiment.
Works in bull markets via Bull Power strength and in bear markets via Bear Power extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(source, length):
    """Exponential Moving Average with proper seed"""
    if length <= 0:
        return np.full_like(source, np.nan)
    result = np.full_like(source, np.nan)
    if len(source) < length:
        return result
    # Seed with SMA
    result[length-1] = np.mean(source[:length])
    # EMA formula
    alpha = 2.0 / (length + 1.0)
    for i in range(length, len(source)):
        result[i] = alpha * source[i] + (1 - alpha) * result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (using 6h close)
    ema13 = ema(close, 13)
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average (20-period)
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup: need EMA13 (13) and daily EMA34 (34) and vol (20)
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        bull = bull_power[i]
        bear = bear_power[i]
        vol_now = volume[i]
        vol_avg = vol_20_1d_aligned[i]
        trend = ema34_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Entry conditions
        if position == 0:
            # Long: Bull Power positive, volume confirmation, uptrend
            if bull > 0 and vol_filter and close[i] > trend:
                signals[i] = size
                position = 1
            # Short: Bear Power negative, volume confirmation, downtrend
            elif bear < 0 and vol_filter and close[i] < trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or trend breaks
            if bull <= 0 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power turns positive or trend breaks
            if bear >= 0 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
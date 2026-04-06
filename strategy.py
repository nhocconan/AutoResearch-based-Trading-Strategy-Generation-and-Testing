#!/usr/bin/env python3
"""
4h Supertrend + Volume Confirmation
Hypothesis: Supertrend captures trend direction effectively. Combined with volume confirmation and ATR stoploss, it provides robust signals in both bull and bear markets. Uses 1d trend filter for higher timeframe context. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14389_4h_supertrend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    uptrend = np.full(n, True)
    
    # Calculate Supertrend
    for i in range(atr_period, n):
        # Upper band
        if i == atr_period:
            upper_band[i] = hl2[i] + multiplier * atr[i]
            lower_band[i] = hl2[i] - multiplier * atr[i]
        else:
            upper_band[i] = hl2[i] + multiplier * atr[i]
            lower_band[i] = hl2[i] - multiplier * atr[i]
            
            # Adjust bands
            if upper_band[i] > upper_band[i-1] or close[i-1] < upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if lower_band[i] < lower_band[i-1] or close[i-1] > lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
        
        # Trend direction
        if i == atr_period:
            uptrend[i] = close[i] > upper_band[i]
        else:
            if close[i] > upper_band[i-1]:
                uptrend[i] = True
            elif close[i] < lower_band[i-1]:
                uptrend[i] = False
            else:
                uptrend[i] = uptrend[i-1]
                if uptrend[i] and lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
                if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
        
        # Supertrend value
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(atr_period, 200) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(supertrend[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Supertrend turns bearish OR stoploss
            if (not uptrend[i] or close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Supertrend turns bullish OR stoploss
            if (uptrend[i] or close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Supertrend direction + 1d trend filter + volume
            long_setup = uptrend[i] and (close[i] > ema200_1d_aligned[i]) and vol_filter[i]
            short_setup = (not uptrend[i]) and (close[i] < ema200_1d_aligned[i]) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
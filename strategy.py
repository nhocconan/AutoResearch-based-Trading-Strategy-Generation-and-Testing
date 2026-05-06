#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for direction and volume breakout for entry
# Long when: price breaks above 2-period high with volume > 1.5x 20-period average AND 12h Supertrend is bullish
# Short when: price breaks below 2-period low with volume > 1.5x 20-period average AND 12h Supertrend is bearish
# Supertrend provides robust trend filtering to avoid counter-trend trades
# Volume confirms breakout strength to filter false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_12hSupertrend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Supertrend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate ATR for Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_period = 10
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend parameters
    multiplier = 3.0
    
    # Basic upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_12h)
    uptrend = np.ones_like(close_12h, dtype=bool)
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close_12h[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Align 12h Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Breakout levels: 2-period high/low
    high_2 = pd.Series(high).rolling(window=2, min_periods=2).max().values
    low_2 = pd.Series(low).rolling(window=2, min_periods=2).min().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(high_2[i]) or np.isnan(low_2[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 2-period high with volume and uptrend
            if close[i] > high_2[i] and volume_filter[i] and uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 2-period low with volume and downtrend
            elif close[i] < low_2[i] and volume_filter[i] and uptrend_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 2-period low or trend turns bearish
            if close[i] < low_2[i] or uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 2-period high or trend turns bullish
            if close[i] > high_2[i] or uptrend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with weekly trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Weekly trend filter ensures trades align with higher timeframe direction.
# Volume confirmation filters low-participation moves. Designed for low frequency in 6h timeframe.
# Works in bull markets (buy bull power dips in uptrend) and bear markets (sell bear power rallies in downtrend).

name = "6h_elder_ray_weekly_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate weekly EMA13 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema13_1w = close_1w.ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Calculate EMA13 on 6x data for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if required data not available
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: close above/below weekly EMA13
        weekly_uptrend = close[i] > ema13_1w_aligned[i]
        weekly_downtrend = close[i] < ema13_1w_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if weekly trend turns down or bear power strengthens
            if not weekly_uptrend or bear_power[i] > -50:  # Bear power less negative = weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if weekly trend turns up or bull power strengthens
            if not weekly_downtrend or bull_power[i] < 50:  # Bull power less positive = weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: weekly uptrend + bull power strong (>100) + volume confirmation
            if weekly_uptrend and bull_power[i] > 100 and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly downtrend + bear power strong (<-100) + volume confirmation
            elif weekly_downtrend and bear_power[i] < -100 and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
Go long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA)
and price > 1d EMA34 (bull regime). Go short when jaws < teeth < lips and price < 1d EMA34 (bear regime).
Volume must exceed 1.5x 20-period average to confirm institutional participation.
Alligator identifies trends, daily trend filters direction, volume confirms strength.
Target: 25-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    n = len(source)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: (prev * (length-1) + current) / length
    for i in range(length, n):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator (SMMA: 13, 8, 5)
    jaws = smma(close, 13)   # Blue line (13-period)
    teeth = smma(close, 8)   # Red line (8-period)
    lips = smma(close, 5)    # Green line (5-period)
    
    # Volume filter: volume > 1.5x average (20-period)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13), volume MA (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Alligator values
        jaw_val = jaws[i]
        tooth_val = teeth[i]
        lip_val = lips[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_ma_val
        
        # Alligator alignment: jaws > teeth > lips (bullish) or jaws < teeth < lips (bearish)
        bullish_alignment = jaw_val > tooth_val > lip_val
        bearish_alignment = jaw_val < tooth_val < lip_val
        
        if position == 0:
            # Bull regime (price > daily EMA34): look for long when bullish alignment + volume
            if ema_trend > 0 and bullish_alignment and vol_filter:
                signals[i] = size
                position = 1
            # Bear regime (price < daily EMA34): look for short when bearish alignment + volume
            elif ema_trend < 0 and bearish_alignment and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator loses bullish alignment or trend changes to bear
            if not bullish_alignment or ema_trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator loses bearish alignment or trend changes to bull
            if not bearish_alignment or ema_trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0
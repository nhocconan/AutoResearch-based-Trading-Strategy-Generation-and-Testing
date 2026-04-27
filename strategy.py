# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

"""
Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions.
- 12h EMA50 defines trend direction (long when price > EMA, short when price < EMA).
- Volume > 1.5x 20-period average confirms momentum.
- Works in both bull/bear markets: mean reversion in range, trend-following in strong moves.
- Target: 50-150 total trades over 4 years (12-37/year).
"""

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(len(high_12h), np.nan)
    lowest_low = np.full(len(low_12h), np.nan)
    
    for i in range(13, len(high_12h)):
        highest_high[i] = np.max(high_12h[i-13:i+1])
        lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    williams_r = np.full(len(high_12h), np.nan)
    mask = denom != 0
    williams_r[mask] = ((highest_high[mask] - close_12h[mask]) / denom[mask]) * -100
    
    # Calculate 12h EMA50
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R, EMA, and volume MA
    start_idx = max(13, ema_period - 1, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > 12h EMA50 + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                price > ema_12h_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 12h EMA50 + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  price < ema_12h_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R > -50 (mean reversion) or trend change
            if (williams_r_aligned[i] > -50 or 
                price < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R < -50 (mean reversion) or trend change
            if (williams_r_aligned[i] < -50 or 
                price > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0
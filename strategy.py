#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA200 trend filter and volume spike
# Williams %R identifies overbought/oversold conditions; in strong trends (price > EMA200),
# we look for oversold bounces (long) and in weak trends (price < EMA200) for overbought reversals (short).
# Volume spike confirms conviction. Works in bull/bear via trend filter.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 200-period EMA
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * (2 / (200 + 1)) + 
                         ema_200[i-1] * (1 - (2 / (200 + 1))))
    
    # Get 1d data for Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period Williams %R
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    williams_r = np.full(len(high_1d), np.nan)
    for i in range(13, len(high_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral if no range
    
    # Align indicators to 12h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R (14), EMA200 (200), volume MA (20)
    start_idx = max(14, 200, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > 1d EMA200 (uptrend) + volume spike
            if (williams_r_aligned[i] < -80 and 
                price > ema_200_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 1d EMA200 (downtrend) + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  price < ema_200_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R > -50 (mean reversion) or trend change
            if (williams_r_aligned[i] > -50 or 
                price < ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R < -50 (mean reversion) or trend change
            if (williams_r_aligned[i] < -50 or 
                price > ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_EMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_ema_pullback_1d_trend_volume_v1
Hypothesis: In trending markets, price pulls back to the 21-period EMA on 4h before resuming trend.
Use daily trend filter (price vs 50-day EMA) and volume confirmation to enter pullbacks.
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell pullbacks in downtrend).
Low trade frequency due to strict trend + pullback + volume alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h EMA for pullback
    ema_21_4h = pd.Series(close).ewm(span=21, min_periods=21).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_21_4h[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 21 EMA
            if close[i] < ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 21 EMA
            if close[i] > ema_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Pullback long: uptrend (price > daily EMA50) + price near 4h EMA21 + volume
            if close[i] > ema_50_1d_aligned[i] and close[i] <= ema_21_4h[i] * 1.005 and volume[i] > vol_ma[i]:
                position = 1
                signals[i] = 0.25
            # Pullback short: downtrend (price < daily EMA50) + price near 4h EMA21 + volume
            elif close[i] < ema_50_1d_aligned[i] and close[i] >= ema_21_4h[i] * 0.995 and volume[i] > vol_ma[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
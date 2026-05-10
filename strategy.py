#!/usr/bin/env python3
"""
4h_RelativeStrengthIndex_1dTrend_VolumeFilter
Hypothesis: RSI mean reversion at 70/30 levels, filtered by 1d trend and volume, works in bull/bear.
In bull: long RSI<30 in uptrend + volume; short RSI>70 in uptrend + volume (fade strength).
In bear: short RSI>70 in downtrend + volume; long RSI<30 in downtrend + volume (fade weakness).
Targets ~30 trades/year via strict RSI levels + trend + volume confluence.
"""

name = "4h_RelativeStrengthIndex_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) + 1d EMA50 (50) + 1d vol avg (20)
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x average 1d volume (scaled)
        vol_4h = volume[i]
        # Scale 1d volume to 4h equivalent (1d = 6x 4h)
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_filter = vol_4h > vol_4h_equiv * 1.5
        
        if position == 0:
            # Long entry: RSI oversold + uptrend + volume
            if rsi[i] < 30 and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought + downtrend + volume
            elif rsi[i] > 70 and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend breaks
            if rsi[i] > 70 or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or trend breaks
            if rsi[i] < 30 or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
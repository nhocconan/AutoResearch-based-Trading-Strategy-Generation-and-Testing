#!/usr/bin/env python3
"""
4h_RSIVolumeTrendFilter
Hypothesis: RSI identifies overbought/oversold conditions within the prevailing trend. Volume confirms momentum. 
Trend is determined by EMA on 1d timeframe. Works in bull/bear markets by trading with the trend and using volume as confirmation.
Targets 20-40 trades/year by requiring trend alignment, volume expansion, and RSI extremes.
"""

name = "4h_RSIVolumeTrendFilter"
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
    
    # Calculate EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate average volume on 1d
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA (34) and volume average (20) and RSI (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x average 1d volume (scaled)
        vol_4h = volume[i]
        # Scale 1d volume to 4h equivalent (1d = 6x 4h)
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_filter = vol_4h > vol_4h_equiv * 1.5
        
        if position == 0:
            # Long entry: price above EMA (uptrend) + RSI < 40 (pullback) + volume participation
            if uptrend_1d and rsi[i] < 40 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price below EMA (downtrend) + RSI > 60 (pullback) + volume participation
            elif downtrend_1d and rsi[i] > 60 and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA or RSI > 60 (overbought)
            if not uptrend_1d or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA or RSI < 40 (oversold)
            if not downtrend_1d or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1h_4H_1D_RSI_Trend_Volume
Hypothesis: 1-hour RSI mean reversion in direction of 4-hour trend and 1-day volume filter.
Uses 4-hour EMA50 for trend direction and 1-day volume spike for confirmation.
RSI(14) < 30 for long in uptrend, RSI(14) > 70 for short in downtrend.
Volume filter: current volume > 2.0 x 20-day average volume.
Target: 60-150 total trades over 4 years (15-37/year) with low frequency due to multiple filters.
Works in bull/bear by following 4h trend and avoiding counter-trend trades.
"""

name = "1h_4H_1D_RSI_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 1-day volume average (20-period)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    # Align to 1h timeframe
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Get price, volume, high, low
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14), 4h EMA50 (50), 1d vol avg (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 2.0 x 20-day average volume
        volume_filter = volume[i] > vol_avg_1d_aligned[i] * 2.0
        
        if position == 0:
            # Long: uptrend (price > 4h EMA50) AND RSI oversold (<30) AND volume spike
            if close[i] > ema_50_4h_aligned[i] and rsi[i] < 30 and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA50) AND RSI overbought (>70) AND volume spike
            elif close[i] < ema_50_4h_aligned[i] and rsi[i] > 70 and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) OR trend turns bearish
            if rsi[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) OR trend turns bullish
            if rsi[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
"""
1h_4h1d_Trend_Follow_With_Volume
Hypothesis: Trend following on 1h using 4h and 1d moving averages for direction,
with volume confirmation on 1h for entry timing. Designed to work in both bull and bear markets
by filtering trades with strong trend alignment and volume confirmation, targeting low trade frequency.
"""

name = "1h_4h1d_Trend_Follow_With_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate 4h EMA21 for trend
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: > 1.3x 20-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Determine trend direction from 4h and 1d EMAs
        long_trend = ema_4h_aligned[i] > ema_4h_aligned[i-1] and ema_1d_aligned[i] > ema_1d_aligned[i-1]
        short_trend = ema_4h_aligned[i] < ema_4h_aligned[i-1] and ema_1d_aligned[i] < ema_1d_aligned[i-1]
        
        if position == 0:
            # LONG: Uptrend on both 4h and 1d with volume confirmation
            if long_trend and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend on both 4h and 1d with volume confirmation
            elif short_trend and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend breaks down on either 4h or 1d
            if not (ema_4h_aligned[i] > ema_4h_aligned[i-1] and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Trend breaks up on either 4h or 1d
            if not (ema_4h_aligned[i] < ema_4h_aligned[i-1] and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance levels. 
Breakouts above R3 or below S3 with 12h EMA trend filter and volume confirmation 
capture strong momentum moves. Works in both bull and bear markets by filtering 
trades with higher timeframe trend. Target: 25-35 trades/year per symbol to avoid 
fee drag while maintaining edge.
"""

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
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
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]
    plow[0] = low[0]
    pclose[0] = close[0]
    
    pivot = (phigh + plow + pclose) / 3
    range_val = phigh - plow
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    uptrend_12h = close > ema_20_12h_aligned
    downtrend_12h = close < ema_20_12h_aligned
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if position == 0:
            # LONG: break above R3, uptrend on 12h, volume confirmation
            if close[i] > r3[i] and uptrend_12h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, downtrend on 12h, volume confirmation
            elif close[i] < s3[i] and downtrend_12h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below pivot or trend reverses
            if close[i] < pivot[i] or not uptrend_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above pivot or trend reverses
            if close[i] > pivot[i] or not downtrend_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
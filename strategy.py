#!/usr/bin/env python3
# 1D_KAMA_1wTrend_Volume
# Hypothesis: Use 1w KAMA for trend direction and 1d KAMA with RSI and volume for entry/exit.
# Long when 1d KAMA crosses above 1d EMA10 in bullish 1w trend with volume confirmation.
# Short when 1d KAMA crosses below 1d EMA10 in bearish 1w trend with volume confirmation.
# Exit when 1d KAMA crosses back below/above 1d EMA10 or trend changes.
# Designed for low trade frequency (<25/year) with strong trend following in both bull and bear markets.

name = "1D_KAMA_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w KAMA for trend filter
    def kama(close_prices, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.abs(np.diff(close_prices)).cumsum()
        volatility[0] = np.abs(np.diff(close_prices)[0]) if len(close_prices) > 1 else 0
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.zeros_like(close_prices)
        kama_vals[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close_prices[i] - kama_vals[i-1])
        return kama_vals

    kama_1w = kama(df_1w['close'].values)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)

    # Get 1d data for KAMA and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for entry signal
    kama_1d = kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Calculate 1d EMA10 for entry/exit
    ema_10_1d = pd.Series(df_1d['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(ema_10_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA crosses above EMA10 + bullish 1w trend (KAMA > EMA10 on 1w) + volume spike
            if (kama_1d_aligned[i] > ema_10_1d_aligned[i] and 
                kama_1d_aligned[i-1] <= ema_10_1d_aligned[i-1] and
                kama_1w_aligned[i] > ema_10_1d_aligned[i] and  # Simplified: using 1d EMA10 as proxy for 1w trend strength
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA crosses below EMA10 + bearish 1w trend (KAMA < EMA10 on 1w) + volume spike
            elif (kama_1d_aligned[i] < ema_10_1d_aligned[i] and 
                  kama_1d_aligned[i-1] >= ema_10_1d_aligned[i-1] and
                  kama_1w_aligned[i] < ema_10_1d_aligned[i] and  # Simplified: using 1d EMA10 as proxy for 1w trend strength
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA crosses below EMA10 or trend changes (KAMA < EMA10 on 1w)
            if (kama_1d_aligned[i] < ema_10_1d_aligned[i] and 
                kama_1d_aligned[i-1] >= ema_10_1d_aligned[i-1]) or \
               (kama_1w_aligned[i] < ema_10_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA crosses above EMA10 or trend changes (KAMA > EMA10 on 1w)
            if (kama_1d_aligned[i] > ema_10_1d_aligned[i] and 
                kama_1d_aligned[i-1] <= ema_10_1d_aligned[i-1]) or \
               (kama_1w_aligned[i] > ema_10_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
#!/usr/bin/env python3
"""
1d_KAMA_Direction_1dTrend_Volume
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat. 
Trades only when KAMA direction aligns with 1-week EMA trend and volume confirms strength. 
Works in bull markets (buy when KAMA turns up in uptrend) and bear markets (sell when KAMA turns down in downtrend).
"""

name = "1d_KAMA_Direction_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values

    # Get 1d data for KAMA calculation (same timeframe)
    df_1d = prices.copy()  # Since we're on 1d timeframe, prices is already 1d data
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio = |change| / sum(|changes|) over lookback period
    def calculate_kama(series, length=10, fast=2, slow=30):
        change = np.abs(np.diff(series, n=length))
        volatility = np.sum(np.abs(np.diff(series)), axis=0)
        # Avoid division by zero
        er = np.zeros_like(series)
        er[length:] = change / (volatility[length:] + 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(series)
        kama[0] = series[0]
        for i in range(1, len(series)):
            kama[i] = kama[i-1] + sc[i] * (series[i] - kama[i-1])
        return kama

    # Calculate KAMA on 1d data
    kama = calculate_kama(close, length=10, fast=2, slow=30)

    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (current > previous) + 1w uptrend + volume spike
            if kama[i] > kama[i-1] and close[i] > ema34_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (current < previous) + 1w downtrend + volume spike
            elif kama[i] < kama[i-1] and close[i] < ema34_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down or 1w trend turns down
            if kama[i] < kama[i-1] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up or 1w trend turns up
            if kama[i] > kama[i-1] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
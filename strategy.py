#2025-06-14T18:00:00
# 12h_KAMA_Direction_1dTrend_Volume
# Hypothesis: KAMA on 12h indicates price direction; confirm with daily EMA trend and volume spike. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Volume spike confirms institutional participation. Works in bull (KAMA up + uptrend) and bear (KAMA down + downtrend).
# Low frequency due to KAMA smoothing and strict volume confirmation.

name = "12h_KAMA_Direction_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # KAMA on 12h
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close_10|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after KAMA warmup
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(kama[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up (close > KAMA) + daily uptrend + volume spike
            if close[i] > kama[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down (close < KAMA) + daily downtrend + volume spike
            elif close[i] < kama[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down OR trend reversal
            if close[i] < kama[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up OR trend reversal
            if close[i] > kama[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
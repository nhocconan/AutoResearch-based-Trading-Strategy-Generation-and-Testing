#!/usr/bin/env python3
# 1D_Camarilla_Pivot_S3_R3_Breakout_WeeklyTrend_Volume
# Hypothesis: Price breaking above Camarilla R3 or below S3 on daily chart, filtered by weekly trend and volume spikes, captures strong momentum moves with low frequency. Works in bull/bear via weekly trend filter. Targets 7-25 trades/year.

name = "1D_Camarilla_Pivot_S3_R3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Daily Camarilla levels (S3, R3)
    # Pivot = (H + L + C)/3
    # Range = H - L
    # S3 = C - Range * 1.1/2
    # R3 = C + Range * 1.1/2
    pivot = (high + low + close) / 3.0
    rng = high - low
    s3 = close - rng * 1.1 / 2.0
    r3 = close + rng * 1.1 / 2.0

    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume filter: >2.0x 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(s3[i]) or np.isnan(r3[i]) or np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + weekly uptrend + volume spike
            if (close[i] > r3[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_50[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + weekly downtrend + volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_50[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below daily VWAP (approx: (H+L+C)/3) or volatility drop
            vwap_today = (high[i] + low[i] + close[i]) / 3.0
            if close[i] < vwap_today or volume[i] < vol_avg_50[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above daily VWAP or volatility drop
            vwap_today = (high[i] + low[i] + close[i]) / 3.0
            if close[i] > vwap_today or volume[i] < vol_avg_50[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
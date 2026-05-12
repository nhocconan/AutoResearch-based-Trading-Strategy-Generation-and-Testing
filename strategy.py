#!/usr/bin/env python3
# 1d_WeeklyPivot_Trend_Filter
# Hypothesis: Trade weekly pivot breakouts on daily timeframe only when aligned with weekly trend (EMA20) and confirmed by volume spike.
# Weekly pivots from weekly timeframe provide institutional reference points.
# Weekly EMA20 filters counter-trend moves; volume spike ensures institutional participation.
# Designed for low turnover: only trade when price breaks key weekly levels with trend and volume confirmation.
# Works in bull (breakouts up in uptrend) and bear (breakdowns in downtrend) markets.

name = "1d_WeeklyPivot_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly pivot levels (using prior week's OHLC)
    # R1 = 2*P - L
    # S1 = 2*P - H
    # P = (H + L + C) / 3
    if len(df_1w) < 2:
        return np.zeros(n)
    ph = df_1w['high'].shift(1).values  # prior week high
    pl = df_1w['low'].shift(1).values   # prior week low
    pc = df_1w['close'].shift(1).values # prior week close
    p = (ph + pl + pc) / 3.0
    r1 = 2 * p - pl
    s1 = 2 * p - ph
    # Align to daily: weekly pivot values are constant through the week
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Volume spike: current > 2.0x average of last 20 days (~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after EMA20 and volume MA warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > weekly R1 + price > weekly EMA20 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < weekly S1 + price < weekly EMA20 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < weekly pivot P or trend breaks
            p_aligned = align_htf_to_ltf(prices, df_1w, p)
            if (close[i] < p_aligned[i] or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > weekly pivot P or trend breaks
            p_aligned = align_htf_to_ltf(prices, df_1w, p)
            if (close[i] > p_aligned[i] or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
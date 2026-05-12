#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 12h timeframe, trade breakouts above daily Camarilla R1 or below S1 only when aligned with 1d trend (EMA50) and confirmed by volume spike.
# Uses daily reference points with 12h execution for lower trade frequency (target: 12-37/year).
# Works in bull (breakouts up in uptrend) and bear (breakdowns in downtrend) markets by following the daily trend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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

    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    if len(df_1d) < 2:
        return np.zeros(n)
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    # Align to 12h: daily Camarilla values are constant through the day
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume spike: current > 2.0x average of last 2 bars (2 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > daily R1 + price > 1d EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < daily S1 + price < 1d EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < daily pivot P or trend breaks
            # Calculate daily pivot P for exit
            pp = (ph + pl + pc) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
            if (close[i] < pp_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > daily pivot P or trend breaks
            pp = (ph + pl + pc) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
            if (close[i] > pp_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
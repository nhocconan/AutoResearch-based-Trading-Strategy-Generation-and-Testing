#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Trade breakouts above 4h Camarilla R1 or below S1 on 1h timeframe when aligned with 4h EMA50 trend and confirmed by volume spike. Uses 4h for signal direction and 1h for precise entry timing to target 15-37 trades/year. Works in both bull and bear markets by following 4h trend direction and exiting at 4h pivot point.
Timeframe: 1h
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for Camarilla levels and EMA50 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    # Calculate 4h Camarilla pivot levels (using prior 4h bar's OHLC)
    ph = df_4h['high'].shift(1).values  # prior 4h high
    pl = df_4h['low'].shift(1).values   # prior 4h low
    pc = df_4h['close'].shift(1).values # prior 4h close
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    # Align to 1h: 4h Camarilla values are constant through the 4h period
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)

    # Get 4h data for EMA50 trend filter ONCE before loop
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Volume spike: current > 2.0x average of last 12 bars (2 hours on 1h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after EMA50 warmup
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > 4h R1 + price > 4h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: close < 4h S1 + price < 4h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < 4h pivot P
            pp = (ph + pl + pc) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: close > 4h pivot P
            pp = (ph + pl + pc) / 3.0
            pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals
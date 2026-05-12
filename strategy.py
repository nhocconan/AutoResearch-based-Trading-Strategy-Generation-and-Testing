#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Strategy: Trade Camarilla R1/S1 breakouts on 4h timeframe aligned with 1d EMA34 trend and volume spike.
# Long when price breaks above R1 with uptrend (price > EMA34) and volume > 2x MA(24).
# Short when price breaks below S1 with downtrend (price < EMA34) and volume > 2x MA(24).
# Exit when price crosses 1d EMA34 or reverses Camarilla level.
# Designed for low turnover (~30-50 trades/year) with trend alignment and volume confirmation.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
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

    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    if len(df_1d) < 2:
        return np.zeros(n)
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    r1 = pc + (ph - pl) * 1.1 / 12.0
    s1 = pc - (ph - pl) * 1.1 / 12.0
    # Align to 4h: daily Camarilla levels are constant through the day
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume spike: current > 2.0x average of last 24 bars (4 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > daily R1 + price > 1d EMA34 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < daily S1 + price < 1d EMA34 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < 1d EMA34 or price < daily S1 (reversal)
            if (close[i] < ema_34_1d_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > 1d EMA34 or price > daily R1 (reversal)
            if (close[i] > ema_34_1d_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
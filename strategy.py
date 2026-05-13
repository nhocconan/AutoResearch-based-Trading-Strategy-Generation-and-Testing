#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Camarilla R3/S3 breakout with volume spike and 1d EMA34 trend filter on 4h timeframe.
# Long when price breaks above R3 with volume spike and close > 1d EMA34; short when breaks below S3 with volume spike and close < 1d EMA34.
# Exit when price crosses back through the central pivot (mean reversion).
# Uses tight entry conditions to limit trades (~30-40/year) and avoid fee drag. Works in bull/bear by capturing breakouts with trend alignment.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Camarilla pivot levels (based on previous day)
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    for i in range(1, n):
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        R3[i] = pc + (ph - pl) * 1.1 / 2
        S3[i] = pc - (ph - pl) * 1.1 / 2
        pivot[i] = (ph + pl + pc) / 3

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(pivot[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 with volume spike and price above 1d EMA34 (uptrend)
            if close[i] > R3[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with volume spike and price below 1d EMA34 (downtrend)
            elif close[i] < S3[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below central pivot (mean reversion)
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above central pivot
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
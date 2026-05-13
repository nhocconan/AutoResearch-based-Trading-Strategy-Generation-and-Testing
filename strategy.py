#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with daily trend filter and volume confirmation.
# Long when price breaks above R1 with volume spike and price above 1d EMA34 (uptrend).
# Short when price breaks below S1 with volume spike and price below 1d EMA34 (downtrend).
# Exit when price retraces to the Camarilla pivot point (mean reversion within range).
# Designed for 15-30 trades/year to minimize fee drag. Works in bull/bear by capturing institutional breakouts aligned with daily trend.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Camarilla levels (based on previous day)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    PP = np.full(n, np.nan)
    for i in range(1, n):
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_ = ph - pl
        PP[i] = (ph + pl + pc) / 3
        R1[i] = pc + (range_ * 1.1 / 12)
        S1[i] = pc - (range_ * 1.1 / 12)

    # Volume confirmation: current volume > 2.0 x 24-period average
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if data is not ready
        if np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(PP[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above daily EMA34 (uptrend)
            if close[i] > R1[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below daily EMA34 (downtrend)
            elif close[i] < S1[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to pivot point (mean reversion)
            if close[i] <= PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to pivot point
            if close[i] >= PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
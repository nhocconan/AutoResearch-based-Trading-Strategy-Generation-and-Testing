#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout with volume confirmation and 12h EMA21 trend filter on 4h timeframe.
# Long: Price breaks above R1 with volume spike and price above 12h EMA21 (uptrend).
# Short: Price breaks below S1 with volume spike and price below 12h EMA21 (downtrend).
# Exit: Price crosses back through the Camarilla midpoint (P).
# Works in bull/bear by capturing breakouts aligned with medium-term trend and volume confirmation.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Previous day's Camarilla levels (R1, S1, P)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous day's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        camarilla_high[i] = pc + 1.1 * (ph - pl) / 12  # R1
        camarilla_low[i] = pc - 1.1 * (ph - pl) / 12   # S1
        camarilla_pivot[i] = (ph + pl + pc) / 3        # P

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    # Get 12h EMA21 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or np.isnan(camarilla_pivot[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above 12h EMA21 (uptrend)
            if close[i] > camarilla_high[i] and volume_spike[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below 12h EMA21 (downtrend)
            elif close[i] < camarilla_low[i] and volume_spike[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot (P)
            if close[i] < camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot (P)
            if close[i] > camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
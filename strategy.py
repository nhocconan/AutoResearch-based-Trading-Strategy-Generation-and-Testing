#!/usr/bin/env python3
# 1h_4H_TRIX_1D_EMA_CROSSOVER
# Hypothesis: TRIX on 4h identifies momentum shifts, confirmed by 1h price crossing 1d EMA50. 
# TRIX > 0 + price above 1d EMA50 = long; TRIX < 0 + price below 1d EMA50 = short. 
# TRIX filters whipsaws, EMA provides trend direction. Works in bull/bear via momentum confirmation.
# Target: 15-37 trades/year via strict TRIX + EMA confluence.

name = "1h_4H_TRIX_1D_EMA_CROSSOVER"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # TRIX (15-period EMA of EMA of EMA)
    close_4h = df_4h['close'].values
    ema1 = pd.Series(close_4h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    trix_4h = np.where(ema3 == 0, 0, trix_raw)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Align 4h TRIX to 1h
    trix_4h_aligned = align_htf_to_ltf(prices, df_4h, trix_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        trix_val = trix_4h_aligned[i]
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]

        if position == 0:
            # LONG: TRIX positive + price above 1d EMA50
            if trix_val > 0 and price > ema_50:
                signals[i] = 0.20
                position = 1
            # SHORT: TRIX negative + price below 1d EMA50
            elif trix_val < 0 and price < ema_50:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative OR price below 1d EMA50
            if trix_val < 0 or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: TRIX turns positive OR price above 1d EMA50
            if trix_val > 0 or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals
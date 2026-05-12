#!/usr/bin/env python3
"""
12h_ParabolicSAR_TF_Signal_System
Hypothesis: Parabolic SAR on 12h captures trend direction and provides trailing stop levels.
Combined with 1d EMA50 trend filter and volume confirmation, it avoids whipsaws in both bull and bear markets.
Long when PSAR flips below price (uptrend) with 1d uptrend and volume spike.
Short when PSAR flips above price (downtrend) with 1d downtrend and volume spike.
Exit when PSAR reverses or 1d trend fails.
"""

name = "12h_ParabolicSAR_TF_Signal_System"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Parabolic SAR (12h timeframe)
    # Initialize
    psar = np.full(n, np.nan)
    bull = True  # True for uptrend
    af = 0.02    # acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # extreme point

    psar[0] = low[0] if bull else high[0]

    for i in range(1, n):
        prev_psar = psar[i-1]
        psar[i] = prev_psar + af * (ep - prev_psar)

        # Ensure PSAR stays within the prior period's range
        if bull:
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])

        # Trend reversal check
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        if np.isnan(psar[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: PSAR flipped below price (uptrend) + 1d uptrend + volume spike
            if bull and close[i] > psar[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: PSAR flipped above price (downtrend) + 1d downtrend + volume spike
            elif not bull and close[i] < psar[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_30[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: PSAR flips above price or 1d trend turns down
            if not bull or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: PSAR flips below price or 1d trend turns up
            if bull or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
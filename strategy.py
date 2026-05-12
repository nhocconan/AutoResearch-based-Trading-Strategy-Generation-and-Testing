#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_1dTrend_Volume
# Hypothesis: TRIX zero cross signals momentum shifts, filtered by 1d EMA34 trend direction.
# Volume spikes confirm institutional participation. Works in bull markets via long entries
# in uptrends and bear markets via short entries in downtrends. Target: 20-30 trades/year.

name = "4h_TRIX_ZeroCross_1dTrend_Volume"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # TRIX (15,9,9) on 4h - triple smoothed ROC
    # Step 1: EMA1 of close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 2: EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 3: EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX: 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value undefined
    # Signal line: EMA9 of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values

    # Volume spike: current > 2.0x average of last 6 bars (1 day)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above signal + 1d EMA34 uptrend + volume spike
            if (trix_raw[i] > trix_signal[i] and 
                trix_raw[i-1] <= trix_signal[i-1] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal + 1d EMA34 downtrend + volume spike
            elif (trix_raw[i] < trix_signal[i] and 
                  trix_raw[i-1] >= trix_signal[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal or trend breaks
            if trix_raw[i] < trix_signal[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal or trend breaks
            if trix_raw[i] > trix_signal[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
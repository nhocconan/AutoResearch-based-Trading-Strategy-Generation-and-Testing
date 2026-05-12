#!/usr/bin/env python3
# 6h_Keltner_Breakout_12hTrend_1dVolumeFilter
# Hypothesis: Keltner channel breakouts with 12h EMA50 trend filter and 1d volume spike filter capture strong trending moves while avoiding false breakouts in chop. Works in both bull and bear by following higher timeframe trend and requiring volume confirmation. Targets 50-150 trades over 4 years.

name = "6h_Keltner_Breakout_12hTrend_1dVolumeFilter"
timeframe = "6h"
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

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)  # 2x average volume
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)

    # Keltner Channel: EMA20 ± 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner upper + 12h EMA50 uptrend + 1d volume spike
            if (close[i] > keltner_upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner lower + 12h EMA50 downtrend + 1d volume spike
            elif (close[i] < keltner_lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 (mean reversion) or trend reversal
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 (mean reversion) or trend reversal
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
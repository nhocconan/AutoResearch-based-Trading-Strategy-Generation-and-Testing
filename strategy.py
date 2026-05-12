#!/usr/bin/env python3
"""
4h_TRIX_Momentum_VolumeSpike_12hTrend
Hypothesis: TRIX momentum (12,9) combined with volume spikes and 12h EMA50 trend filter captures strong directional moves while avoiding whipsaws. Works in bull (momentum continuation) and bear (sharp reversals) markets by filtering with higher timeframe trend.
"""

name = "4h_TRIX_Momentum_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate TRIX: EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # percentage
    trix = trix_raw.fillna(0).values

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX positive + rising + volume spike + 12h uptrend
            if trix[i] > 0 and trix[i] > trix[i-1] and volume[i] > vol_avg_20[i] * 2 and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: TRIX negative + falling + volume spike + 12h downtrend
            elif trix[i] < 0 and trix[i] < trix[i-1] and volume[i] > vol_avg_20[i] * 2 and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative or 12h trend turns down
            if trix[i] < 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: TRIX turns positive or 12h trend turns up
            if trix[i] > 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
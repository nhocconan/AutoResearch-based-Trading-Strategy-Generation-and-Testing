#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v2
Refined version to reduce trade frequency and improve robustness:
- Require volume > 1.5x 20-period average (was 1.3x) to reduce false breakouts
- Add ATR(14) filter: require ATR > 0.5 * 20-period ATR average to avoid low-volatility chops
- Exit on close crossing 10-period EMA of closing price (faster exit than Donchian touch)
- Position size: 0.25
"""

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # ATR(14) filter: avoid low-volatility chop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_avg_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values

    # Exit EMA: 10-period EMA of close
    ema10_close = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(atr[i]) or np.isnan(atr_avg_20[i]) or np.isnan(ema10_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + 12h uptrend + volume spike + vol filter
            if (close[i] > high_max_20[i-1] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                atr[i] > atr_avg_20[i] * 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 12h downtrend + volume spike + vol filter
            elif (close[i] < low_min_20[i-1] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  atr[i] > atr_avg_20[i] * 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 10-period EMA or trend turns down
            if close[i] < ema10_close[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 10-period EMA or trend turns up
            if close[i] > ema10_close[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
#!/usr/bin/env python3
# 4h_Keltner_Breakout_Trend_Filter
# Hypothesis: Trade Keltner Channel breakouts filtered by 1-day EMA trend and volume spikes.
# Long when price breaks above upper Keltner (EMA20 + 2*ATR) during 1-day uptrend with volume > 1.5x average.
# Short when price breaks below lower Keltner (EMA20 - 2*ATR) during 1-day downtrend with volume spike.
# Exit when price crosses back below/above EMA20 or opposite breakout occurs.
# Designed for low trade frequency (<30/year) with strong trend capture in both bull and bear markets.

name = "4h_Keltner_Breakout_Trend_Filter"
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

    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate ATR(20) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Calculate EMA20 for Keltner mid-line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper = ema_20 + 2 * atr_20
    keltner_lower = ema_20 - 2 * atr_20

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Keltner in 1-day uptrend + volume spike
            if close[i] > keltner_upper[i] and close[i-1] <= keltner_upper[i-1]:
                if ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below lower Keltner in 1-day downtrend + volume spike
            elif close[i] < keltner_lower[i] and close[i-1] >= keltner_lower[i-1]:
                if ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20 or breaks below lower Keltner
            if close[i] < ema_20[i] or close[i] < keltner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20 or breaks above upper Keltner
            if close[i] > ema_20[i] or close[i] > keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
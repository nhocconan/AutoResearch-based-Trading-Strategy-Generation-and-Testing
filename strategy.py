#!/usr/bin/env python3
# 12h_Keltner_Breakout_1dTrend_Volume
# Hypothesis: Keltner Channel breakouts capture trend moves with fewer whipsaws than Bollinger Bands.
# Enter long when price breaks above upper Keltner band with volume spike and 1d EMA uptrend.
# Enter short when price breaks below lower Keltner band with volume spike and 1d EMA downtrend.
# Exit when price crosses back to EMA(20) to avoid missed reversals.
# Designed for 12h timeframe with 1d trend filter to reduce trade frequency and improve win rate.
# Target: 15-25 trades/year per symbol.

name = "12h_Keltner_Breakout_1dTrend_Volume"
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
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate EMA(20) and ATR(10) for Keltner Channel
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values

    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    upper_keltner = ema_20 + (2.0 * atr_10)
    lower_keltner = ema_20 - (2.0 * atr_10)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner with volume spike and 1d EMA uptrend
            if close[i] > upper_keltner[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner with volume spike and 1d EMA downtrend
            elif close[i] < lower_keltner[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA(20) (mean reversion signal)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA(20) (mean reversion signal)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
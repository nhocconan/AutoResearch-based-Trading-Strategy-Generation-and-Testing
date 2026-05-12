#!/usr/bin/env python3
# 160119: 6h_Keltner_Channel_Reversal_12hTrend_Volume
# Hypothesis: Mean reversion at Keltner Channel bands with 12h trend filter and volume confirmation.
# Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
# Uses 6h timeframe with 12h EMA trend filter and 1.5x ATR Keltner bands for dynamic support/resistance.
# Volume spike (>1.5x 20-period average) confirms momentum behind the reversal.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_Keltner_Channel_Reversal_12hTrend_Volume"
timeframe = "6h"
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

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate 12-period ATR for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=12, min_periods=12).mean().values

    # Calculate EMA(20) for middle band
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channel bands: ±1.5 * ATR from EMA(20)
    kc_upper = ema_20 + 1.5 * atr
    kc_lower = ema_20 - 1.5 * atr

    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after warmup period
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower Keltner band + 12h uptrend + volume confirmation
            if (low[i] <= kc_lower[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper Keltner band + 12h downtrend + volume confirmation
            elif (high[i] >= kc_upper[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above EMA(20) (mean reversion complete)
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA(20) (mean reversion complete)
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
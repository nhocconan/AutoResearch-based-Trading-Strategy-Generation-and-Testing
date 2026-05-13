#!/usr/bin/env python3
# 4h_Pullback_Breakout_Volume
# Hypothesis: Pullback to EMA20 then breakout in direction of 1d trend with volume confirmation.
# Works in bull/bear by following 1d trend. Entry: price retests EMA20 then breaks out with volume > 1.5x average.
# Exit: reverse signal or trailing stop via EMA50 cross.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Pullback_Breakout_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # EMA20 for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # EMA50 for exit signal
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above EMA20, pulls back to touch EMA20, then breaks up with volume
            if (close[i] > ema20[i] and 
                low[i] <= ema20[i] * 1.005 and  # allow small tolerance for pullback
                close[i] > ema20[i-1] and      # breakout above prior EMA20
                ema50_1d_aligned[i] < close[i] and  # uptrend filter
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below EMA20, bounces to touch EMA20, then breaks down with volume
            elif (close[i] < ema20[i] and 
                  high[i] >= ema20[i] * 0.995 and  # allow small tolerance for bounce
                  close[i] < ema20[i-1] and        # breakdown below prior EMA20
                  ema50_1d_aligned[i] > close[i] and  # downtrend filter
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below EMA50 or reverse signal
            if close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above EMA50 or reverse signal
            if close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
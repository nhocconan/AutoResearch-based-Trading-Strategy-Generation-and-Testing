#!/usr/bin/env python3
"""
6h_Momentum_With_Volume_Regime_Filter
Hypothesis: On 6h timeframe, enter long when price breaks above 6h high with volume >1.5x average and RSI < 70 (not overbought), enter short when price breaks below 6h low with volume >1.5x average and RSI > 30 (not oversold). Use 12h EMA50 as trend filter to avoid counter-trend trades. Designed to capture momentum bursts in both bull and bear markets while avoiding exhaustion moves. Targets 20-40 trades per year to minimize fee drag.
"""

name = "6h_Momentum_With_Volume_Regime_Filter"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # RSI(14) for momentum exhaustion filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 6h high + volume spike + not overbought + 12h uptrend
            if (close[i] > high[i-1] and  # broke previous 6h bar high
                volume[i] > vol_avg_20[i] * 1.5 and
                rsi[i] < 70 and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 6h low + volume spike + not oversold + 12h downtrend
            elif (close[i] < low[i-1] and  # broke previous 6h bar low
                  volume[i] > vol_avg_20[i] * 1.5 and
                  rsi[i] > 30 and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 6h low OR RSI overbought OR trend turns down
            if (close[i] < low[i-1] or 
                rsi[i] > 70 or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 6h high OR RSI oversold OR trend turns up
            if (close[i] > high[i-1] or 
                rsi[i] < 30 or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
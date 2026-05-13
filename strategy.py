#!/usr/bin/env python3
# 4h_RSI_Trend_Filter_Simple
# Hypothesis: Use RSI(14) > 50 for long bias and < 50 for short bias on 4h, combined with 1d trend filter (price > 200 EMA = long bias, < = short bias) and volume confirmation (>1.5x 20-period average).
# Trend filter from higher timeframe reduces false signals in choppy markets. Works in bull (follows long signals with bullish 1d trend) and bear (avoids longs in bearish 1d trend, takes shorts).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_RSI_Trend_Filter_Simple"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)

    # RSI(14) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50 (bullish momentum) + price above 1d EMA200 (bullish trend) + volume spike
            if (rsi[i] > 50 and 
                close[i] > ema_200_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 (bearish momentum) + price below 1d EMA200 (bearish trend) + volume spike
            elif (rsi[i] < 50 and 
                  close[i] < ema_200_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 50 or price below 1d EMA200
            if (rsi[i] < 50 or close[i] < ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 50 or price above 1d EMA200
            if (rsi[i] > 50 or close[i] > ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
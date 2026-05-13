#!/usr/bin/env python3
# 4h_RSI_MeanReversion_1dTrend_Volume
# Hypothesis: Use RSI extreme reversals on 4h with 1d trend filter and volume confirmation.
# In bull markets (price > 1d EMA50), go long when RSI(14) < 30 (oversold) with volume spike.
# In bear markets (price < 1d EMA50), go short when RSI(14) > 70 (overbought) with volume spike.
# This targets mean reversion within the trend, reducing false signals.
# Volume spike confirms momentum behind the reversal.
# Target: 80-150 total trades over 4 years = 20-38/year.

name = "4h_RSI_MeanReversion_1dTrend_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate RSI (14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above 1d EMA50 (bullish trend) + RSI oversold + volume spike
            if (close[i] > ema_50_1d_aligned[i] and 
                rsi[i] < 30 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price below 1d EMA50 (bearish trend) + RSI overbought + volume spike
            elif (close[i] < ema_50_1d_aligned[i] and 
                  rsi[i] > 70 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below 1d EMA50 or RSI overbought (exit reversion)
            if (close[i] < ema_50_1d_aligned[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above 1d EMA50 or RSI oversold (exit reversion)
            if (close[i] > ema_50_1d_aligned[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
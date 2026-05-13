#!/usr/bin/env python3
# 12h_RSI_MeanReversion_with_Volume_Confirmation
# Hypothesis: RSI(14) oversold (<30) or overbought (>70) with volume confirmation (>1.5x 30-bar average) signals mean reversion.
# Uses 1w EMA50 as trend filter: only take long when price > EMA50 (uptrend bias), short when price < EMA50 (downtrend bias).
# Designed for 12h timeframe to capture multi-day mean reversion moves in both bull and bear markets.
# Target: 15-30 trades/year to avoid fee drag.

name = "12h_RSI_MeanReversion_with_Volume_Confirmation"
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

    # RSI calculation
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        gain = np.zeros_like(close_prices)
        loss = np.zeros_like(close_prices)
        gain[period:] = pd.Series(up).rolling(window=period, min_periods=period).mean().values[period-1:]
        loss[period:] = pd.Series(down).rolling(window=period, min_periods=period).mean().values[period-1:]
        rs = np.where(loss != 0, gain / loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals

    rsi_vals = rsi(close, 14)

    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_vals[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) + price > EMA50 (uptrend filter) + volume spike
            if (rsi_vals[i] < 30 and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + price < EMA50 (downtrend filter) + volume spike
            elif (rsi_vals[i] > 70 and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete) or price < EMA50 (trend change)
            if (rsi_vals[i] > 50 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (mean reversion complete) or price > EMA50 (trend change)
            if (rsi_vals[i] < 50 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
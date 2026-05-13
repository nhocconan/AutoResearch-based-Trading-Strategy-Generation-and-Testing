#!/usr/bin/env python3
# 4h_Small_Range_Breakout_1dTrend_Volume
# Hypothesis: After small-range periods (ATR-based volatility contraction), breakouts in the direction of the daily trend
# with volume confirmation capture momentum. The daily trend filter (price vs EMA50) aligns with higher timeframe momentum,
# while volume surge confirms institutional participation. Works in bull and bear markets by following daily trend.

name = "4h_Small_Range_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Daily ATR(14) for volatility measurement
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr1 = np.maximum(tr1, np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)

    # Daily range (high - low)
    range_1d = high_1d - low_1d
    range_1d_aligned = align_htf_to_ltf(prices, df_1d, range_1d)

    # Volatility contraction: today's range < 0.5 * ATR(14)
    vol_contract = range_1d_aligned < 0.5 * atr14_1d_aligned

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(range_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + volatility contraction + breakout above high + volume spike
            if (close[i] > ema50_1d_aligned[i] and 
                vol_contract[i] and 
                close[i] > high[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + volatility contraction + breakdown below low + volume spike
            elif (close[i] < ema50_1d_aligned[i] and 
                  vol_contract[i] and 
                  close[i] < low[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below low or trend turns bearish
            if close[i] < low[i-1] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above high or trend turns bullish
            if close[i] > high[i-1] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
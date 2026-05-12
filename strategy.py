#!/usr/bin/env python3
"""
12h_ThreeCandlePattern_1dTrend
Hypothesis: Three consecutive bullish/bearish candles on 12h timeframe with volume confirmation indicates strong momentum. 
In bull markets, long on 3 consecutive bullish closes with volume above average. 
In bear markets, short on 3 consecutive bearish closes with volume above average.
Uses 1d EMA50 for trend filter to avoid counter-trend trades. Works in both bull and bear markets.
"""

name = "12h_ThreeCandlePattern_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for 3 consecutive bullish/bearish candles
            bullish_3 = (close[i] > open_price[i]) and (close[i-1] > open_price[i-1]) and (close[i-2] > open_price[i-2])
            bearish_3 = (close[i] < open_price[i]) and (close[i-1] < open_price[i-1]) and (close[i-2] < open_price[i-2])
            
            # LONG: 3 bullish closes + above average volume + 1d uptrend
            if bullish_3 and volume[i] > vol_avg_20[i] * 1.5 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 3 bearish closes + above average volume + 1d downtrend
            elif bearish_3 and volume[i] > vol_avg_20[i] * 1.5 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or momentum loss
            if close[i] < ema50_1d_aligned[i] or not (close[i] > open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or momentum loss
            if close[i] > ema50_1d_aligned[i] or not (close[i] < open_price[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
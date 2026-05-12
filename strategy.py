#!/usr/bin/env python3
"""
4h_Keltner_CCI_Bounce_1dTrend_VolumeFilter
Hypothesis: Uses Keltner Channel mean reversion combined with CCI momentum and daily trend filter.
Buys near lower Keltner band when CCI indicates oversold and daily trend is bullish.
Sells near upper Keltner band when CCI indicates overbought and daily trend is bearish.
Designed for 4h timeframe to capture mean-reversion moves with low trade frequency.
Works in both bull and bear markets by adapting to daily trend context.
"""

name = "4h_Keltner_CCI_Bounce_1dTrend_VolumeFilter"
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

    # Get daily data for trend filter (call once before loop)
    df_d = get_htf_data(prices, '1d')
    if len(d_d) < 20:
        return np.zeros(n)

    # Calculate daily EMA20 for trend filter
    close_d = pd.Series(df_d['close'].values)
    ema20_d = close_d.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channel (20, 1.5) - using 20-period EMA and ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.abs(high - low)).rolling(window=20, min_periods=20).mean().values
    keltner_upper = ema20 + 1.5 * atr
    keltner_lower = ema20 - 1.5 * atr

    # CCI (20) - Commodity Channel Index
    typical_price = (high + low + close) / 3
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mean_dev = pd.Series(np.abs(typical_price - ma_tp)).rolling(window=20, min_periods=20).mean().values
    # Avoid division by zero
    cci = np.where(mean_dev != 0, (typical_price - ma_tp) / (0.015 * mean_dev), 0)

    # Volume confirmation: 4-period average (half day of 4h data)
    vol_avg_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start from 20 to have enough data for indicators
        # Get aligned values for current 4h bar
        ema20_d_aligned = align_htf_to_ltf(prices, df_d, ema20_d)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_d_aligned) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(cci[i]) or np.isnan(vol_avg_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near lower Keltner band + CCI oversold + bullish daily trend + volume
            if (close[i] <= keltner_lower[i] * 1.01 and  # Near lower band
                cci[i] < -100 and  # Oversold
                close[i] > ema20_d_aligned and  # Bullish daily trend
                volume[i] > vol_avg_4[i] * 1.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Price near upper Keltner band + CCI overbought + bearish daily trend + volume
            elif (close[i] >= keltner_upper[i] * 0.99 and  # Near upper band
                  cci[i] > 100 and  # Overbought
                  close[i] < ema20_d_aligned and  # Bearish daily trend
                  volume[i] > vol_avg_4[i] * 1.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches middle Keltner band or CCI turns negative
            if (close[i] >= ema20[i] or cci[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches middle Keltner band or CCI turns positive
            if (close[i] <= ema20[i] or cci[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
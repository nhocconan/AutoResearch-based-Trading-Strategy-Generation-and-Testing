#!/usr/bin/env python3
"""
4h_RSI_Extreme_Reversion_With_Volume
Hypothesis: Extreme RSI readings (RSI<20 or RSI>80) with volume confirmation and 1d trend filter.
Mean reversion works in both bull and bear markets when price is overextended.
Volume confirms institutional interest in the move. Trend filter avoids fighting strong trends.
Designed for ~20-30 trades/year to stay within fee limits.
"""

name = "4h_RSI_Extreme_Reversion_With_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate RSI (14) with proper min_periods
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Get aligned values for current 4h bar (only uses completed daily bars)
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(vol_threshold_val) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Extreme oversold (RSI<20) + volume spike (2x) + 1d uptrend
            if (rsi[i] < 20 and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Extreme overbought (RSI>80) + volume spike (2x) + 1d downtrend
            elif (rsi[i] > 80 and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (above 40) or trend fails
            if rsi[i] > 40 or close[i] < ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (below 60) or trend fails
            if rsi[i] < 60 or close[i] > ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
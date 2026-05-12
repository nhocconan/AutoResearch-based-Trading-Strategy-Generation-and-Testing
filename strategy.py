#!/usr/bin/env python3
# 12h_VWAP_Reversion_1wTrend_Filter
# Hypothesis: Mean reversion to VWAP with 1-week trend filter to avoid counter-trend trades.
# Uses VWAP deviation on 12h timeframe with 1-week EMA trend filter and volume confirmation.
# Designed to work in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend) markets.
# Target: 12-30 trades/year on 12h timeframe to stay under fee drag threshold.

name = "12h_VWAP_Reversion_1wTrend_Filter"
timeframe = "12h"
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

    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate VWAP on 12h timeframe (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator

    # Calculate VWAP deviation as percentage
    vwap_dev = (close - vwap) / vwap * 100.0

    # Calculate volume filter (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vwap_dev[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price below VWAP (oversold) in uptrend with volume confirmation
            if (vwap_dev[i] < -0.5 and  # Price below VWAP by more than 0.5%
                close[i] > ema50_1w_aligned[i] and  # Uptrend filter
                volume[i] > volume_filter[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: price above VWAP (overbought) in downtrend with volume confirmation
            elif (vwap_dev[i] > 0.5 and  # Price above VWAP by more than 0.5%
                  close[i] < ema50_1w_aligned[i] and  # Downtrend filter
                  volume[i] > volume_filter[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to VWAP or trend weakens
            if (vwap_dev[i] > 0.0 or  # Price back to or above VWAP
                close[i] < ema50_1w_aligned[i]):  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to VWAP or trend weakens
            if (vwap_dev[i] < 0.0 or  # Price back to or below VWAP
                close[i] > ema50_1w_aligned[i]):  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
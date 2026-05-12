#!/usr/bin/env python3
"""
12h_VWAP_Deviation_1dTrend_VolumeFilter
Hypothesis: Price deviations from 1d VWAP (Volume Weighted Average Price) combined with 1d EMA trend filter and volume confirmation captures mean-reversion in range markets and trend continuation in trending markets. Works in both bull and bear by following 1d trend direction while using VWAP as dynamic support/resistance. Targets 15-25 trades/year on 12h timeframe to minimize fee drag.
"""

name = "12h_VWAP_Deviation_1dTrend_VolumeFilter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d VWAP: typical price * volume / cumulative volume
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vp_1d = typical_price_1d * df_1d['volume'].values
    cum_vp_1d = np.nancumsum(vp_1d)
    cum_vol_1d = np.nancumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_vp_1d, cum_vol_1d, out=np.full_like(cum_vp_1d, np.nan), where=cum_vol_1d!=0)

    # Shift by 1 to use previous day's VWAP
    prev_vwap_1d = np.roll(vwap_1d, 1)
    prev_vwap_1d[0] = np.nan

    # Calculate standard deviation of price from VWAP over 20 days
    price_dev_1d = typical_price_1d - vwap_1d
    # Rolling std dev of price deviation
    price_dev_series = pd.Series(price_dev_1d)
    std_dev_20d = price_dev_series.rolling(window=20, min_periods=20).std().values
    std_dev_20d[0:19] = np.nan  # Ensure proper warmup

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align all 1d indicators to 12h timeframe
    prev_vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_vwap_1d)
    std_dev_20d_aligned = align_htf_to_ltf(prices, df_1d, std_dev_20d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.3x 20-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(prev_vwap_1d_aligned[i]) or np.isnan(std_dev_20d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below VWAP - 1.5*std dev + 1d EMA50 uptrend + volume filter
            if (close[i] < (prev_vwap_1d_aligned[i] - 1.5 * std_dev_20d_aligned[i]) and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP + 1.5*std dev + 1d EMA50 downtrend + volume filter
            elif (close[i] > (prev_vwap_1d_aligned[i] + 1.5 * std_dev_20d_aligned[i]) and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP (mean reversion complete)
            if close[i] > prev_vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP (mean reversion complete)
            if close[i] < prev_vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
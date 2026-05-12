#!/usr/bin/env python3
"""
4h_VWAP_Reversion_with_1dTrend_Filter
Hypothesis: Mean-reversion to VWAP on 4h with trend filter from 1d EMA34.
Long when price crosses below VWAP (oversold) and 1d EMA34 trending up.
Short when price crosses above VWAP (overbought) and 1d EMA34 trending down.
Uses VWAP deviation with Bollinger Bands (20,2) to avoid whipsaws.
Targets 20-40 trades/year to minimize fee drag and work in both bull/bear markets.
"""

name = "4h_VWAP_Reversion_with_1dTrend_Filter"
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
    typical_price = (high + low + close) / 3

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate VWAP (cumulative typical price * volume / cumulative volume)
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.zeros_like(cum_tpv), where=cum_vol!=0)

    # VWAP deviation bands (20-period, 2 std dev)
    vwap_dev = typical_price - vwap
    vwap_ma = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    upper_band = vwap + vwap_ma + 2 * vwap_std
    lower_band = vwap + vwap_ma - 2 * vwap_std

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(vwap[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower band AND 1d uptrend
            if close[i] <= lower_band[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper band AND 1d downtrend
            elif close[i] >= upper_band[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP OR trend turns down
            if close[i] >= vwap[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP OR trend turns up
            if close[i] <= vwap[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
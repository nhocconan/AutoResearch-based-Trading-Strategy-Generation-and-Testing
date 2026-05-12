#!/usr/bin/env python3
# 12h_Volume_Weighted_CCI_Trend
# Hypothesis: CCI (Commodity Channel Index) identifies overbought/oversold conditions,
# while volume-weighted price confirms institutional participation. Daily trend filter
# ensures alignment with higher timeframe momentum. Designed for low trade frequency
# (<25/year) to minimize fee drag in 12h timeframe, targeting 50-100 trades over 4 years.

name = "12h_Volume_Weighted_CCI_Trend"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate CCI(20): Typical Price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - sma_tp) / (0.015 * mad)
    cci_values = cci.values

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume-weighted price close: VWAP approximation for trend confirmation
    # Using close * volume to weight price by volume
    price_volume = close * volume
    pv_series = pd.Series(price_volume)
    vol_series = pd.Series(volume)
    vwp = (pv_series.rolling(window=20, min_periods=20).sum() / 
           vol_series.rolling(window=20, min_periods=20).sum())  # Volume-weighted price
    vwp_values = vwp.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after CCI needs 20 bars
        # Skip if any required data is NaN
        if (np.isnan(cci_values[i]) or np.isnan(cci_values[i-1]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vwp_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CCI crosses above -100 (oversold bounce) with volume confirmation and uptrend
            if (cci_values[i-1] <= -100 and cci_values[i] > -100 and
                close[i] > vwp_values[i] and  # Price above volume-weighted price
                close[i] > ema34_1d_aligned[i]):  # Above daily EMA trend
                signals[i] = 0.25
                position = 1
            # SHORT: CCI crosses below +100 (overbought rejection) with volume confirmation and downtrend
            elif (cci_values[i-1] >= 100 and cci_values[i] < 100 and
                  close[i] < vwp_values[i] and  # Price below volume-weighted price
                  close[i] < ema34_1d_aligned[i]):  # Below daily EMA trend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI crosses below +100 (overbought) or price breaks volume-weighted support
            if cci_values[i] < 100 or close[i] < vwp_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI crosses above -100 (oversold) or price breaks volume-weighted resistance
            if cci_values[i] > -100 or close[i] > vwp_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
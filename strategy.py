#!/usr/bin/env python3
# 6h_VWAP_Reversion_1wTrend_Filter
# Hypothesis: On 6h timeframe, long when price deviates >1.5σ below weekly VWAP and weekly trend is up (price > weekly EMA50); short when price deviates >1.5σ above weekly VWAP and weekly trend is down. Uses volume-weighted price to identify mean reversion extremes in trending markets, reducing false signals. Targets 20-40 trades/year to minimize fee drag. Works in bull/bear by aligning with weekly trend.

name = "6h_VWAP_Reversion_1wTrend_Filter"
timeframe = "6h"
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
    typical_price = (high + low + close) / 3.0

    # Get weekly data for VWAP and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0

    # Calculate weekly VWAP: cumulative TP*V / cumulative V
    cum_tpv = np.cumsum(typical_price_1w * volume_1w)
    cum_vol = np.cumsum(volume_1w)
    vwap_1w = np.where(cum_vol != 0, cum_tpv / cum_vol, 0.0)

    # Weekly VWAP standard deviation: sqrt( sum((TP-VWAP)^2 * V) / sum(V) )
    squared_dev = (typical_price_1w - vwap_1w) ** 2 * volume_1w
    cum_squared_dev = np.cumsum(squared_dev)
    vwap_var_1w = np.where(cum_vol != 0, cum_squared_dev / cum_vol, 0.0)
    vwap_std_1w = np.sqrt(np.maximum(vwap_var_1w, 0))

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align weekly indicators to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    vwap_std_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_std_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(vwap_std_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > 1.5σ below VWAP + weekly uptrend
            if (close[i] < (vwap_1w_aligned[i] - 1.5 * vwap_std_1w_aligned[i]) and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price > 1.5σ above VWAP + weekly downtrend
            elif (close[i] > (vwap_1w_aligned[i] + 1.5 * vwap_std_1w_aligned[i]) and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend breaks down
            if close[i] >= vwap_1w_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend breaks up
            if close[i] <= vwap_1w_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
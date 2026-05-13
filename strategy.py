# MULTI-TIMEFRAME VWAP TREND FILTER
# Hypothesis: Use VWAP from higher timeframe (1w/1d) as dynamic support/resistance.
# Long when price crosses above weekly VWAP with volume spike and daily VWAP uptrend.
# Short when price crosses below weekly VWAP with volume spike and daily VWAP downtrend.
# Exit when price crosses back below/above daily VWAP.
# VWAP adapts to market regime and provides institutional-grade levels.
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "6h_VWAP_Trend_Filter"
timeframe = "6h"
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

    # Calculate VWAP on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly VWAP: cumulative typical price * volume / cumulative volume
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    
    # Calculate VWAP on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values

    # Align VWAPs to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)

    # Volume confirmation: current volume > 1.8 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: cross above weekly VWAP with volume spike and daily VWAP uptrend
            if close[i] > vwap_1w_aligned[i] and close[i-1] <= vwap_1w_aligned[i-1] and volume_spike[i] and close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: cross below weekly VWAP with volume spike and daily VWAP downtrend
            elif close[i] < vwap_1w_aligned[i] and close[i-1] >= vwap_1w_aligned[i-1] and volume_spike[i] and close[i] < vwap_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below daily VWAP (mean reversion)
            if close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above daily VWAP
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
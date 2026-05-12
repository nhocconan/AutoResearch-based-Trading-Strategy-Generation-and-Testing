#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout
Hypothesis: Uses weekly pivot levels to identify key support/resistance zones.
Enters long when price breaks above weekly R1 with bullish weekly trend (price > weekly VWAP),
enters short when price breaks below weekly S1 with bearish weekly trend (price < weekly VWAP).
Uses volume confirmation to avoid false breakouts. Designed for 6h timeframe to capture
multi-day moves while avoiding excessive trading. Weekly pivot provides structural levels
that work in both bull and bear markets by adapting to current price regime.
"""

name = "6h_WeeklyPivot_RangeBreakout"
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

    # Get weekly data for pivot points and trend filter (call once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)

    # Calculate weekly VWAP for trend filter
    typical_price_w = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    vwap_w = (typical_price_w * df_w['volume']).cumsum() / df_w['volume'].cumsum()
    vwap_w = vwap_w.values

    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    hh_w = df_w['high'].values
    ll_w = df_w['low'].values
    cc_w = df_w['close'].values
    
    pivot_w = (hh_w + ll_w + cc_w) / 3
    r1_w = 2 * pivot_w - ll_w
    s1_w = 2 * pivot_w - hh_w

    # Volume confirmation: 24-period average (4 days of 6h data)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # Start from 24 to have enough data for volume average
        # Get aligned values for current 6h bar
        vwap_w_aligned = align_htf_to_ltf(prices, df_w, vwap_w)[i]
        r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)[i]
        s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)[i]
        vol_avg_val = vol_avg_24[i]
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_w_aligned) or np.isnan(r1_w_aligned) or np.isnan(s1_w_aligned) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly R1 with bullish weekly trend and volume surge
            if (close[i] > r1_w_aligned and 
                close[i] > vwap_w_aligned and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 with bearish weekly trend and volume surge
            elif (close[i] < s1_w_aligned and 
                  close[i] < vwap_w_aligned and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below weekly VWAP or breaks below S1 (reversal signal)
            if (close[i] < vwap_w_aligned or close[i] < s1_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above weekly VWAP or breaks above R1 (reversal signal)
            if (close[i] > vwap_w_aligned or close[i] > r1_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
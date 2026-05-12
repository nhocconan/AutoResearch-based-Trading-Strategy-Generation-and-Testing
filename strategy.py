#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries,
confirmed by weekly trend (price above/below weekly VWAP) and volume surge.
Designed for 6h timeframe to capture multi-day moves with low trade frequency.
Works in both bull and bear markets by adapting to weekly trend context.
"""

name = "6h_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
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

    # Get daily data for Camarilla pivot points (call once before loop)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 5:
        return np.zeros(n)

    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)

    # Calculate daily Camarilla pivot levels (standard formula)
    # Based on previous day's high, low, close
    hh_d = df_d['high'].values
    ll_d = df_d['low'].values
    cc_d = df_d['close'].values

    # Camarilla levels
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # R2 = C + (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # S2 = C - (H-L)*1.1/6
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    r3_d = cc_d + (hh_d - ll_d) * 1.1 / 4
    s3_d = cc_d - (hh_d - ll_d) * 1.1 / 4

    # Calculate weekly VWAP for trend filter
    typical_price_w = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    vwap_w = (typical_price_w * df_w['volume']).cumsum() / df_w['volume'].cumsum()
    vwap_w = vwap_w.values

    # Volume confirmation: 12-period average (3 days of 6h data)
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(12, n):  # Start from 12 to have enough data for volume average
        # Get aligned values for current 6h bar
        r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)[i]
        s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)[i]
        vwap_w_aligned = align_htf_to_ltf(prices, df_w, vwap_w)[i]
        vol_avg_val = vol_avg_12[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r3_d_aligned) or np.isnan(s3_d_aligned) or 
            np.isnan(vwap_w_aligned) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above daily R3 with bullish weekly trend and volume surge
            if (close[i] > r3_d_aligned and 
                close[i] > vwap_w_aligned and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below daily S3 with bearish weekly trend and volume surge
            elif (close[i] < s3_d_aligned and 
                  close[i] < vwap_w_aligned and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below daily S3 or weekly VWAP (reversal signal)
            if (close[i] < s3_d_aligned or close[i] < vwap_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above daily R3 or weekly VWAP (reversal signal)
            if (close[i] > r3_d_aligned or close[i] > vwap_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
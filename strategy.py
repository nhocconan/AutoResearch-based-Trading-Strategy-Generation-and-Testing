#!/usr/bin/env python3
# 6h_Volume_Weighted_RSI_Trend_Follow
# Hypothesis: Use 6h RSI with volume-weighted smoothing to reduce noise, combined with 1d trend filter.
# Long when VW-RSI crosses above 50 in 1d uptrend, short when crosses below 50 in 1d downtrend.
# Volume weighting makes RSI more responsive to institutional flow, reducing false signals in chop.
# Works in bull (buy strength on volume) and bear (sell weakness on volume).

name = "6h_Volume_Weighted_RSI_Trend_Follow"
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
    volume = prices['volume'].values

    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume-weighted RSI (14 period)
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted RS
    avg_vg = pd.Series(vol_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vl = pd.Series(vol_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vl != 0, avg_vg / avg_vl, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if np.isnan(vw_rsi[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VW-RSI crosses above 50 + 1d uptrend
            if vw_rsi[i] > 50 and vw_rsi[i-1] <= 50 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VW-RSI crosses below 50 + 1d downtrend
            elif vw_rsi[i] < 50 and vw_rsi[i-1] >= 50 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VW-RSI crosses below 50 OR trend reversal
            if vw_rsi[i] < 50 and vw_rsi[i-1] >= 50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VW-RSI crosses above 50 OR trend reversal
            if vw_rsi[i] > 50 and vw_rsi[i-1] <= 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
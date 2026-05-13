#!/usr/bin/env python3
# 4h_DeMark_TD_Setup_1dTrend_Regime
# Hypothesis: Use Tom DeMark's TD Sequential setup (count 9) for reversal signals, filtered by 1d EMA50 trend and Choppiness Index regime. Long on TD Setup 9 in downtrend during ranging markets; short on TD Setup 9 in uptrend during ranging markets. TD Sequential identifies exhaustion points; trend filter ensures alignment with higher timeframe momentum; chop filter avoids false signals in strong trends. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Low frequency due to strict setup count requirement.

name = "4h_DeMark_TD_Setup_1dTrend_Regime"
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

    # Get daily data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Choppiness Index on daily for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr1 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Absolute price change over 14 periods
    price_change = np.abs(close_1d - np.roll(close_1d, 14))
    price_change[0:14] = np.nan
    chop = 100 * np.log10(sum_tr / (price_change + atr1 * 14)) / np.log10(10)
    chop[0:14] = np.nan
    # Chop > 61.8 = ranging, < 38.2 = trending
    chop_range = chop > 61.8
    
    # Align daily indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    # TD Sequential setup phase (simplified: count 9 for exhaustion)
    # Setup: 9 consecutive closes higher/lower than 4 periods ago
    # Bullish setup: close > close 4 periods ago for 9 consecutive periods
    # Bearish setup: close < close 4 periods ago for 9 consecutive periods
    close_shifted = np.roll(close, 4)
    close_shifted[0:4] = np.nan
    higher_than_4ago = close > close_shifted
    lower_than_4ago = close < close_shifted
    
    # Count consecutive higher/lower
    consec_higher = np.zeros(n)
    consec_lower = np.zeros(n)
    for i in range(1, n):
        if higher_than_4ago[i]:
            consec_higher[i] = consec_higher[i-1] + 1
        else:
            consec_higher[i] = 0
        if lower_than_4ago[i]:
            consec_lower[i] = consec_lower[i-1] + 1
        else:
            consec_lower[i] = 0
    
    # TD Setup 9 signals
    td_setup_9_long = consec_higher >= 9  # Potential exhaustion, look for reversal down
    td_setup_9_short = consec_lower >= 9  # Potential exhaustion, look for reversal up
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(chop_range_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TD Setup 9 (exhaustion of downtrend) + ranging market + price above daily EMA50 (uptrend bias)
            if td_setup_9_long[i] and chop_range_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TD Setup 9 (exhaustion of uptrend) + ranging market + price below daily EMA50 (downtrend bias)
            elif td_setup_9_short[i] and chop_range_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TD Setup 9 completion (count 13) or trend change or chop regime ends
            if td_setup_9_long[i] and consec_higher[i] >= 13 or close[i] < ema50_1d_aligned[i] or not chop_range_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TD Setup 9 completion (count 13) or trend change or chop regime ends
            if td_setup_9_short[i] and consec_lower[i] >= 13 or close[i] > ema50_1d_aligned[i] or not chop_range_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
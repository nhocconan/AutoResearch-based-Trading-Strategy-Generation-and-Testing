#!/usr/bin/env python3
# 6H_WILLIAMS_R_MULTIPLIER_1D_TREND_FILTER
# Hypothesis: Williams %R measures overbought/oversold conditions on the daily chart.
# In 1d uptrend (price > EMA50), go long when Williams %R crosses above -80 from below.
# In 1d downtrend (price < EMA50), go short when Williams %R crosses below -20 from above.
# This captures mean reversion within the trend, working in both bull and bear markets.
# Uses Williams %R(14) on daily timeframe with trend filter to avoid counter-trend trades.
# Target: 20-30 trades/year on 6h timeframe.

name = "6H_WILLIAMS_R_MULTIPLIER_1D_TREND_FILTER"
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
    
    # Daily data for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = (high_14 - close_1d) / (high_14 - low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((high_14 - low_14) == 0, -50, williams_r)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r_aligned, 1)
    williams_r_prev[0] = 50  # neutral value
    
    # Cross above -80 (oversold to normal)
    cross_above_80 = (williams_r_aligned > -80) & (williams_r_prev <= -80)
    # Cross below -20 (overbought to normal)
    cross_below_20 = (williams_r_aligned < -20) & (williams_r_prev >= -20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + Williams %R crosses above -80
            if (close[i] > ema50_aligned[i] and cross_above_80[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + Williams %R crosses below -20
            elif (close[i] < ema50_aligned[i] and cross_below_20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or overbought condition
            if (close[i] <= ema50_aligned[i] or williams_r_aligned[i] >= -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or oversold condition
            if (close[i] >= ema50_aligned[i] or williams_r_aligned[i] <= -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
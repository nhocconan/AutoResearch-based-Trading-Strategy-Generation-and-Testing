#!/usr/bin/env python3
# 6h_Weekly_Pivot_Pullback_Entry_1dTrend_Volume
# Hypothesis: On 6h timeframe, enter long when price pulls back to weekly pivot S1/S2 in a weekly uptrend, short when price pulls back to weekly pivot R1/R2 in a weekly downtrend, with daily trend confirmation and volume spike to avoid false signals. Weekly pivot provides institutional support/resistance, pullback offers better risk-reward than breakout. Daily trend filter ensures alignment with intermediate trend, volume reduces noise. Designed for low frequency (~20-40 trades/year) to minimize fee drag in bear markets.

name = "6h_Weekly_Pivot_Pullback_Entry_1dTrend_Volume"
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
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pp = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pp - low_w
    s1 = 2 * pp - high_w
    r2 = pp + (high_w - low_w)
    s2 = pp - (high_w - low_w)
    r3 = high_w + 2 * (pp - low_w)
    s3 = low_w - 2 * (high_w - pp)
    
    # Weekly trend: price above/below weekly pivot
    trend_w_up = close_w > pp
    trend_w_down = close_w < pp
    
    # Align weekly data to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    trend_w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_w_up.astype(float))
    trend_w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_w_down.astype(float))
    
    # Daily trend: EMA34 on daily close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(trend_w_up_aligned[i]) or
            np.isnan(trend_w_down_aligned[i]) or np.isnan(trend_1d_up_aligned[i]) or
            np.isnan(trend_1d_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: pullback to weekly S1/S2 in weekly uptrend with daily uptrend and volume
            if ((low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and
                trend_w_up_aligned[i] > 0.5 and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: pullback to weekly R1/R2 in weekly downtrend with daily downtrend and volume
            elif ((high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and
                  trend_w_down_aligned[i] > 0.5 and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to weekly pivot or weekly trend fails
            if (close[i] >= pp_aligned[i] or
                trend_w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to weekly pivot or weekly trend fails
            if (close[i] <= pp_aligned[i] or
                trend_w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
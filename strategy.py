#!/usr/bin/env python3
# 1d_Weekly_Pivot_Breakout_Momentum
# Hypothesis: Breakout of weekly pivot levels (R1/S1) with daily trend filter and volume confirmation.
# Uses weekly pivot levels for structural support/resistance, daily EMA50 for trend bias, and volume spike for confirmation.
# Designed to work in both bull and bear markets by trading breakouts aligned with higher timeframe trend.
# Targets 15-25 trades/year to minimize fee drag.

name = "1d_Weekly_Pivot_Breakout_Momentum"
timeframe = "1d"
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
    
    # Weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Weekly data for pivot levels (using previous week's OHLC)
    close_1w_arr = df_1w['close'].values
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_close = np.concatenate([[close_1w_arr[0]], close_1w_arr[:-1]])
    prev_high = np.concatenate([[high_1w_arr[0]], high_1w_arr[:-1]])
    prev_low = np.concatenate([[low_1w_arr[0]], low_1w_arr[:-1]])
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    weekly_r1 = (2 * pivot) - prev_low
    weekly_s1 = (2 * pivot) - prev_high
    
    # Align weekly pivot levels to daily
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: price breaks above weekly R1 with weekly uptrend and volume spike
            if (close[i] > weekly_r1_aligned[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with weekly downtrend and volume spike
            elif (close[i] < weekly_s1_aligned[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly pivot or trend fails
            if (close[i] < pivot[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly pivot or trend fails
            if (close[i] > pivot[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
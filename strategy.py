#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Daily breakout of weekly Camarilla R1/S1 levels with weekly trend filter (EMA20).
# Weekly trend determines bias, daily price triggers entry/exit. Designed to work in both bull and bear markets by
# only taking breakouts aligned with higher timeframe trend. Target: 10-25 trades/year to minimize fee drag.

name = "1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter"
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
    
    # 1w data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema20_1w
    trend_1w_down = close_1w < ema20_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Weekly OHLC for Camarilla levels (using previous week's values)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_close = np.concatenate([[close_1w[0]], close_1w[:-1]])
    prev_high = np.concatenate([[high_1w[0]], high_1w[:-1]])
    prev_low = np.concatenate([[low_1w[0]], low_1w[:-1]])
    
    # Weekly Camarilla levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to daily
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R1 with weekly uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S1 with weekly downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly Camarilla pivot or trend fails
            if (close[i] < camarilla_pivot_aligned[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly Camarilla pivot or trend fails
            if (close[i] > camarilla_pivot_aligned[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
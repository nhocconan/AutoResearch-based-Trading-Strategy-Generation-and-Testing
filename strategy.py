#!/usr/bin/env python3
"""
12H_Donchian_20_Breakout_1dTrend_Volume
Hypothesis: Breakouts from 20-period Donchian Channel with 1-day trend filter and volume confirmation.
Works in bull markets by following the daily trend; in bear markets, the trend filter prevents counter-trend entries.
Target: 15-25 trades/year per symbol. Low frequency reduces fee drag.
"""
name = "12H_Donchian_20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_upper = high_series.rolling(window=20, min_periods=20).max().values
    dc_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d trend (EMA50) - using daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        trend_up = trend_1d_up_aligned[i] > 0.5
        trend_down = trend_1d_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: breakout above upper Donchian + 1d uptrend + volume
            if close[i] > dc_upper[i] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below lower Donchian + 1d downtrend + volume
            elif close[i] < dc_lower[i] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price closes back below the Donchian middle (mean reversion)
            dc_middle = (dc_upper[i] + dc_lower[i]) / 2
            if close[i] < dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes back above the Donchian middle
            dc_middle = (dc_upper[i] + dc_lower[i]) / 2
            if close[i] > dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
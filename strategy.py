#!/usr/bin/env python3
# 4h_ThreeBar_Range_Breakout_1dTrend_Volume
# Hypothesis: Three-bar range breakouts capture low-volatility breakouts with momentum.
# Uses 1d trend filter and volume confirmation to avoid false breakouts.
# Works in bull markets via breakouts and in bear via mean-reversion at extremes.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_ThreeBar_Range_Breakout_1dTrend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Three-bar range: min(low) and max(high) of last 3 bars
    low_min = np.minimum.reduce([low, np.roll(low, 1), np.roll(low, 2)])
    high_max = np.maximum.reduce([high, np.roll(high, 1), np.roll(high, 2)])
    
    # Shift to avoid look-ahead: use previous bar's three-bar range
    low_min_prev = np.roll(low_min, 1)
    high_max_prev = np.roll(high_max, 1)
    low_min_prev[0:2] = np.nan
    high_max_prev[0:2] = np.nan
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(low_min_prev[i]) or np.isnan(high_max_prev[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above 3-bar high with 1d uptrend and volume spike
            if (close[i] > high_max_prev[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below 3-bar low with 1d downtrend and volume spike
            elif (close[i] < low_min_prev[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below 3-bar low or trend fails
            if (close[i] < low_min_prev[i] or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above 3-bar high or trend fails
            if (close[i] > high_max_prev[i] or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
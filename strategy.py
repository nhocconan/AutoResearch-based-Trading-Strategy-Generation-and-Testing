#!/usr/bin/env python3
# 12h_Chaikin_Money_Flow_1dTrend_Volume
# Hypothesis: Use Chaikin Money Flow (CMF) on 12h for momentum, filtered by 1d EMA50 trend and volume confirmation.
# CMF > 0 indicates buying pressure, < 0 selling pressure. Trades only when CMF crosses zero with trend and volume alignment.
# Designed for low trade frequency (~20-40/year) to minimize fee drift and work in both bull/bear markets via trend filter.

name = "12h_Chaikin_Money_Flow_1dTrend_Volume"
timeframe = "12h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 12h Chaikin Money Flow (20-period)
    # CMF = sum((Close - Low - (High - Close)) / (High - Low) * Volume) / sum(Volume) over period
    # Avoid division by zero: where high == low, use 0
    hl_range = high - low
    # Where hl_range == 0, set money flow multiplier to 0 (no range)
    mf_multiplier = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    mf_volume = mf_multiplier * volume
    
    # Sum over 20 periods
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    # Avoid division by zero
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0.0)
    
    # 12h volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(cmf[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: CMF crosses above zero with uptrend and volume
            if cmf[i] > 0 and cmf[i-1] <= 0 and trend_1d_up_aligned[i] > 0.5 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below zero with downtrend and volume
            elif cmf[i] < 0 and cmf[i-1] >= 0 and trend_1d_down_aligned[i] > 0.5 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CMF crosses below zero or trend fails
            if cmf[i] < 0 or trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CMF crosses above zero or trend fails
            if cmf[i] > 0 or trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
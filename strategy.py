#!/usr/bin/env python3
# 6h_AwesomeOscillator_1wTrend_Confirm
# Hypothesis: Awesome Oscillator (AO) on 6h signals momentum shifts, filtered by 1-week EMA50 trend direction.
# In bull markets, only take long signals; in bear markets, only take short signals.
# Uses volume confirmation (volume > 1.5x 20-bar average) to avoid false signals.
# Targets 20-35 trades/year to minimize fee drag. Works in both regimes by aligning with higher timeframe trend.
# Combines momentum (AO) with trend filter (1w EMA) for robust performance.

name = "6h_AwesomeOscillator_1wTrend_Confirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Awesome Oscillator: (5-period SMA of median price) - (34-period SMA of median price)
    median_price = (high + low) / 2
    sma5 = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    sma34 = pd.Series(median_price).rolling(window=34, min_periods=34).mean().values
    ao = sma5 - sma34
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(ao[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: AO crosses above zero with weekly uptrend and volume
            if (ao[i] > 0 and ao[i-1] <= 0 and
                trend_1w_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: AO crosses below zero with weekly downtrend and volume
            elif (ao[i] < 0 and ao[i-1] >= 0 and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: AO crosses back below zero or trend fails
            if (ao[i] < 0 or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: AO crosses back above zero or trend fails
            if (ao[i] > 0 or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
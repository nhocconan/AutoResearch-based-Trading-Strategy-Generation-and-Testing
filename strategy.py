#!/usr/bin/env python3
# 1d_Weekly_Trend_Breakout
# Hypothesis: Weekly trend (EMA34) filters direction; daily Donchian breakout (20) with volume spike captures momentum.
# Works in bull (breakouts with trend) and bear (mean reversion at Donchian bands with trend filter).
# Low trade frequency (~10-25/year) avoids fee drag.

name = "1d_Weekly_Trend_Breakout"
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
    
    # Weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current > 2.0 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above Donchian high with weekly uptrend and volume spike
            if (close[i] > donchian_high[i] and 
                trend_1w_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with weekly downtrend and volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_1w_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below Donchian low or trend fails
            if (close[i] < donchian_low[i] or 
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above Donchian high or trend fails
            if (close[i] > donchian_high[i] or 
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
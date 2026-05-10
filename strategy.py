#!/usr/bin/env python3
# 4H_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long when: price breaks above 20-period high + volume > 1.5x average + 1d uptrend.
# Short when: price breaks below 20-period low + volume > 1.5x average + 1d downtrend.
# Exit when: price crosses the 10-period moving average (opposite direction).
# Trend filter prevents counter-trend trades in choppy markets.
# Target: 20-40 trades/year per symbol. Works in bull/bear by following 1d trend.

name = "4H_Donchian_Breakout_Volume_TrendFilter"
timeframe = "4h"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 10-period MA for exit
    close_series = pd.Series(close)
    ma10 = close_series.rolling(window=10, min_periods=10).mean().values
    
    # 1d trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ma10[i]) or
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
            # Enter long: breakout above Donchian high + volume + 1d uptrend
            if close[i] > donchian_high[i] and volume_confirm and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below Donchian low + volume + 1d downtrend
            elif close[i] < donchian_low[i] and volume_confirm and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below 10-period MA
            if close[i] < ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above 10-period MA
            if close[i] > ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
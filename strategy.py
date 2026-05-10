#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Trend_Filter
# Hypothesis: In 2025+ bear/range markets, breakouts from Donchian channels with volume confirmation and trend filter capture strong directional moves while avoiding false breakouts. Uses 1-day EMA50 for trend filter and Donchian(20) for breakout signals. Designed for low trade frequency (~20-50/year) to minimize fee drag and work in both bull (follows breakouts) and bear (avoids counter-trend).

name = "12h_Donchian_Breakout_Volume_Trend_Filter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h chart
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), EMA50_1d (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout detection
        if i > 0:
            breakout_high = (close[i] > donchian_high[i-1]) and (close[i-1] <= donchian_high[i-1])
            breakdown_low = (close[i] < donchian_low[i-1]) and (close[i-1] >= donchian_low[i-1])
        else:
            breakout_high = False
            breakdown_low = False
        
        if position == 0:
            # Long entry: uptrend + breakout above Donchian high + volume
            if uptrend and breakout_high and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + breakdown below Donchian low + volume
            elif downtrend and breakdown_low and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or reversal breakdown
            if not uptrend or breakdown_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or reversal breakout
            if not downtrend or breakout_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
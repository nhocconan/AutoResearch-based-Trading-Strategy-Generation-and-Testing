#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Uses 1d Donchian channels for breakout signals: long when price breaks above upper band, short when below lower band
# 1w EMA50 determines higher timeframe trend: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation requires current volume > 1.8x 20-period average to filter weak breakouts
# Works in bull/bear: EMA filter ensures we trade with the weekly trend, avoiding counter-trend whipsaws
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian(20) channels
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
        else:
            upper_channel[i] = np.max(high[i-20:i])
            lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < lower Donchian channel OR price < 1w EMA50 (trend change)
            if close[i] < lower_channel[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > upper Donchian channel OR price > 1w EMA50 (trend change)
            if close[i] > upper_channel[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout
            if volume_confirmed:
                # Long entry: price > upper Donchian channel AND price > 1w EMA50 (breakout + uptrend)
                if close[i] > upper_channel[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < lower Donchian channel AND price < 1w EMA50 (breakdown + downtrend)
                elif close[i] < lower_channel[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
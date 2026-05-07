#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1d close > EMA50 (uptrend) AND volume spike.
# Short when price breaks below Donchian lower band AND 1d close < EMA50 (downtrend) AND volume spike.
# Uses Donchian channel for breakout momentum, 1d EMA50 for trend direction, and volume to confirm strength.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and work in both bull and bear markets.
name = "12h_Donchian20_1dTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 12h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian breakout signals
    long_breakout = close > highest_high_20
    short_breakout = close < lowest_low_20
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_1d > ema_50_1d
    trend_down = close_1d < ema_50_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA (higher threshold for lower frequency)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian upper breakout + 1d uptrend + volume spike
            long_condition = long_breakout[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Donchian lower breakout + 1d downtrend + volume spike
            short_condition = short_breakout[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian middle (10-period median) or 1d trend turns down
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < donchian_middle or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price crosses above Donchian middle or 1d trend turns up
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > donchian_middle or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period high AND weekly close > EMA50 (uptrend) AND volume spike.
# Short when price breaks below 20-period low AND weekly close < EMA50 (downtrend) AND volume spike.
# Uses Donchian channels for breakout signals, weekly EMA50 for trend direction, and volume for confirmation.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via breakouts in uptrend and in bear markets via breakdowns in downtrend.
name = "12h_Donchian20_WeeklyTrend_Volume"
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
    
    # Donchian channels (20-period) on 12h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_1w > ema_50_1w
    trend_down = close_1w < ema_50_1w
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above 20-period high + weekly uptrend + volume spike
            long_condition = (close[i] > highest_high_20[i]) and trend_up_aligned[i] and volume_spike[i]
            # Short: Donchian breakdown below 20-period low + weekly downtrend + volume spike
            short_condition = (close[i] < lowest_low_20[i]) and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below 10-period low or weekly trend turns down
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if (close[i] < lowest_low_10) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above 10-period high or weekly trend turns up
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if (close[i] > highest_high_10) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly close > weekly EMA50 (uptrend) AND volume spike.
# Short when price breaks below 20-day low AND weekly close < weekly EMA50 (downtrend) AND volume spike.
# This combines a classic breakout strategy with weekly trend alignment to avoid counter-trend trades.
# Volume confirmation ensures momentum behind the breakout.
# Designed for low frequency (target: 10-25 trades/year) to minimize fee drag and improve generalization.
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakouts in downtrend.
name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on daily data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout signals: price crosses above/below Donchian bands
    breakout_long = (close > highest_high_20) & (np.roll(close, 1) <= highest_high_20)
    breakout_short = (close < lowest_low_20) & (np.roll(close, 1) >= lowest_low_20)
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up_1w = close_1w > ema_50_1w
    trend_down_1w = close_1w < ema_50_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Volume confirmation: current volume > 2.0 * 50-period EMA
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(trend_up_1w_aligned[i]) or 
            np.isnan(trend_down_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + weekly uptrend + volume spike
            long_condition = breakout_long[i] and trend_up_1w_aligned[i] and volume_spike[i]
            # Short: Donchian breakout down + weekly downtrend + volume spike
            short_condition = breakout_short[i] and trend_down_1w_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below 20-day low OR weekly trend turns down
            if (close[i] < lowest_low_20[i] and np.roll(close, 1)[i] >= lowest_low_20[i]) or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above 20-day high OR weekly trend turns up
            if (close[i] > highest_high_20[i] and np.roll(close, 1)[i] <= highest_high_20[i]) or not trend_down_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
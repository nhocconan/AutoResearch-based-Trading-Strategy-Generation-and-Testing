#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR stop.
# Long when price breaks above Donchian upper (20) AND 1d close > EMA50 (uptrend) AND volume spike.
# Short when price breaks below Donchian lower (20) AND 1d close < EMA50 (downtrend) AND volume spike.
# Uses Donchian for breakout, 1d EMA50 for trend, and volume to confirm momentum.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag.
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakdowns in downtrend.
name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian breakout signals
    breakout_up = (high > highest_high_20) & (np.roll(high, 1) <= highest_high_20)
    breakout_down = (low < lowest_low_20) & (np.roll(low, 1) >= lowest_low_20)
    
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
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + 1d uptrend + volume spike
            long_condition = breakout_up[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Donchian breakout down + 1d downtrend + volume spike
            short_condition = breakout_down[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian lower (20) or 1d trend turns down
            if close[i] < lowest_low_20[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian upper (20) or 1d trend turns up
            if close[i] > highest_high_20[i] or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
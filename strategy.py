#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Long when price breaks above 20-period high AND 4h close > EMA50 (uptrend) AND volume spike.
# Short when price breaks below 20-period low AND 4h close < EMA50 (downtrend) AND volume spike.
# Uses Donchian for breakout momentum, 4h EMA50 for trend filter, volume to confirm strength.
# Designed for moderate trade frequency (target: 15-35/year) with session filter (08-20 UTC) to reduce noise.
# Works in bull via long breakouts in uptrend, bear via short breakouts in downtrend.
name = "1h_DonchianBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout signals: price crosses above/below Donchian bands
    breakout_long = (close > highest_high_20) & (np.roll(close, 1) <= highest_high_20)
    breakout_short = (close < lowest_low_20) & (np.roll(close, 1) >= lowest_low_20)
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_4h > ema_50_4h
    trend_down = close_4h < ema_50_4h
    trend_up_aligned = align_htf_to_ltf(prices, df_4h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_4h, trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + 4h uptrend + volume spike + session
            long_condition = breakout_long[i] and trend_up_aligned[i] and volume_spike[i] and session_mask[i]
            # Short: Donchian breakout down + 4h downtrend + volume spike + session
            short_condition = breakout_short[i] and trend_down_aligned[i] and volume_spike[i] and session_mask[i]
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low OR 4h trend turns down
            if (close < lowest_low_20) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above Donchian high OR 4h trend turns up
            if (close > highest_high_20) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
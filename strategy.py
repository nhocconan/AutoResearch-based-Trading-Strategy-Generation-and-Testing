#!/usr/bin/env python3
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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6-week pivot points (using 6-week high/low/close from daily data)
    # We'll use 30-day lookback for 6-week approximation
    lookback_period = 30
    high_6w = pd.Series(high_1d).rolling(window=lookback_period, min_periods=lookback_period).max().values
    low_6w = pd.Series(low_1d).rolling(window=lookback_period, min_periods=lookback_period).min().values
    close_6w = pd.Series(close_1d).rolling(window=lookback_period, min_periods=lookback_period).last().values
    
    # Calculate pivot point and resistance/support levels
    pivot_6w = (high_6w + low_6w + close_6w) / 3
    r1_6w = 2 * pivot_6w - low_6w
    s1_6w = 2 * pivot_6w - high_6w
    r2_6w = pivot_6w + (high_6w - low_6w)
    s2_6w = pivot_6w - (high_6w - low_6w)
    r3_6w = high_6w + 2 * (pivot_6w - low_6w)
    s3_6w = low_6w - 2 * (high_6w - pivot_6w)
    
    # Align 6-week levels to 6h timeframe
    pivot_6w_aligned = align_htf_to_ltf(prices, df_1d, pivot_6w)
    r1_6w_aligned = align_htf_to_ltf(prices, df_1d, r1_6w)
    s1_6w_aligned = align_htf_to_ltf(prices, df_1d, s1_6w)
    r2_6w_aligned = align_htf_to_ltf(prices, df_1d, r2_6w)
    s2_6w_aligned = align_htf_to_ltf(prices, df_1d, s2_6w)
    r3_6w_aligned = align_htf_to_ltf(prices, df_1d, r3_6w)
    s3_6w_aligned = align_htf_to_ltf(prices, df_1d, s3_6w)
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pivot_6w_aligned[i]) or
            np.isnan(r1_6w_aligned[i]) or
            np.isnan(s1_6w_aligned[i]) or
            np.isnan(r2_6w_aligned[i]) or
            np.isnan(s2_6w_aligned[i]) or
            np.isnan(r3_6w_aligned[i]) or
            np.isnan(s3_6w_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current volume above average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > vol_ma_6h[i] * 1.2
        
        # Fade at R3/S3 levels
        fade_short = close[i] >= r3_6w_aligned[i] * 0.995  # Near R3
        fade_long = close[i] <= s3_6w_aligned[i] * 1.005   # Near S3
        
        # Breakout at R4/S4 levels (using R2/S2 as base for simplicity)
        breakout_up = close[i] > r2_6w_aligned[i] * 1.01
        breakout_down = close[i] < s2_6w_aligned[i] * 0.99
        
        # Long conditions: near S3 (fade) OR breakout above R2 with trend + volume
        long_condition = (fade_long and price_above_ema and volume_filter) or \
                         (breakout_up and price_above_ema and volume_filter)
        
        # Short conditions: near R3 (fade) OR breakdown below S2 with trend + volume
        short_condition = (fade_short and price_below_ema and volume_filter) or \
                          (breakout_down and price_below_ema and volume_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or at opposite extreme
        elif position == 1 and (not price_above_ema or close[i] >= r1_6w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or close[i] <= s1_6w_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_6WeekPivot_FadeBreakout_EMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0
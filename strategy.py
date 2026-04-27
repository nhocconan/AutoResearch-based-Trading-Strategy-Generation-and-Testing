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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data for entry timing
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    vol_6h = df_6h['volume'].values
    
    # Calculate 6-week high/low (42 periods of 6h = 1 week)
    week_high_42 = pd.Series(high_6h).rolling(window=42, min_periods=42).max().values
    week_low_42 = pd.Series(low_6h).rolling(window=42, min_periods=42).min().values
    week_high_aligned = align_htf_to_ltf(prices, df_6h, week_high_42)
    week_low_aligned = align_htf_to_ltf(prices, df_6h, week_low_42)
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(week_high_aligned[i]) or 
            np.isnan(week_low_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 6h volume above average (strict)
        volume_filter = vol_ma_6h_aligned[i] > 0 and volume[i] > vol_ma_6h_aligned[i] * 1.5
        
        # Breakout signals: price breaks 6-week high/low
        breakout_up = close[i] > week_high_aligned[i]
        breakout_down = close[i] < week_low_aligned[i]
        
        # Long conditions: bullish trend + volume + upward breakout
        long_condition = (price_above_ema and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: bearish trend + volume + downward breakout
        short_condition = (price_below_ema and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
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

name = "1d_EMA34_6WeekBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0
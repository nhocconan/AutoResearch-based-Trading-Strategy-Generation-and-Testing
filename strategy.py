#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channel (20 periods)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Calculate weekly EMA 50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly volume moving average (20 periods)
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_aligned[i]) or 
            np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: current volume above weekly average
        volume_filter = volume[i] > vol_ma_20_1w_aligned[i]
        
        # Long conditions: price breaks above weekly Donchian high + trend + volume
        long_breakout = close[i] > highest_high_20_aligned[i]
        long_condition = long_breakout and price_above_ema and volume_filter
        
        # Short conditions: price breaks below weekly Donchian low + trend + volume
        short_breakout = close[i] < lowest_low_20_aligned[i]
        short_condition = short_breakout and price_below_ema and volume_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or opposite breakout
        elif position == 1 and (not price_above_ema or close[i] < lowest_low_20_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or close[i] > highest_high_20_aligned[i]):
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

name = "1d_WeeklyDonchian20_Breakout_EMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0
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
    
    # Get weekly data for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-period Donchian channels on weekly
    high_20 = np.full(len(close_1w), np.nan)
    low_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        high_20[i] = np.max(high_1w[i-20:i])
        low_20[i] = np.min(low_1w[i-20:i])
    
    # Calculate 50-period EMA on weekly (trend filter)
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume ratio (current week vs 20-week average)
    volume_ma_20 = np.full(len(volume_1w), np.nan)
    for i in range(20, len(volume_1w)):
        volume_ma_20[i] = np.mean(volume_1w[i-20:i])
    volume_ratio = volume_1w / volume_ma_20
    
    # Align indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-week average
        volume_confirm = volume_ratio_aligned[i] > 1.5
        
        # Entry conditions: breakout in direction of trend with volume
        long_entry = long_breakout and above_ema and volume_confirm
        short_entry = short_breakout and below_ema and volume_confirm
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_ema50_volume_breakout"
timeframe = "1d"
leverage = 1.0
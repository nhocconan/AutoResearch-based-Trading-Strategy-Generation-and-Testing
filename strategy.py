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
    
    # Get weekly data for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly
    high_20_1w = np.full(len(close_1w), np.nan)
    low_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        high_20_1w[i] = np.max(high_1w[i-20:i])
        low_20_1w[i] = np.min(low_1w[i-20:i])
    
    # Calculate 20-period EMA on weekly (trend filter)
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period SMA on weekly (additional trend confirmation)
    sma_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20_1w[i] = np.mean(close_1w[i-20:i])
    
    # Align indicators to 6h timeframe
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Calculate volume moving average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_1w_aligned[i]) or 
            np.isnan(low_20_1w_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/both EMA20 and SMA20 on weekly
        above_both = close[i] > ema_20_1w_aligned[i] and close[i] > sma_20_1w_aligned[i]
        below_both = close[i] < ema_20_1w_aligned[i] and close[i] < sma_20_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_1w_aligned[i]
        short_breakout = close[i] < low_20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = long_breakout and above_both and volume_confirm
        short_entry = short_breakout and below_both and volume_confirm
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_both)
        exit_short = position == -1 and (long_breakout or above_both)
        
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

name = "6h_1w_donchian_ema20_volume_filter"
timeframe = "6h"
leverage = 1.0
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
    
    # Get 1d data for daily high/low (for Donchian channel)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # 20-day Donchian channel (highest high and lowest low)
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1d volume for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / volume_ma
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # Calculate 12h ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    atr_multiplier = 2.5
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume confirmation
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        vol_confirm = volume_ratio_aligned[i] > 1.5  # Volume 1.5x average
        
        long_entry = breakout_long and vol_confirm
        short_entry = breakout_short and vol_confirm
        
        # Exit conditions: ATR-based stop loss or opposite breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Long position: stop if price drops below entry - ATR*multiplier
            # or if price breaks below Donchian low
            exit_long = (close[i] < donchian_low_aligned[i])  # Exit on opposite breakout
        elif position == -1:
            # Short position: stop if price rises above entry + ATR*multiplier
            # or if price breaks above Donchian high
            exit_short = (close[i] > donchian_high_aligned[i])  # Exit on opposite breakout
        
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

name = "12h_1d_donchian_breakout_volume"
timeframe = "12h"
leverage = 1.0
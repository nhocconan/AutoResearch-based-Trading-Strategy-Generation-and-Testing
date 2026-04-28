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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian channel (20-period) - price channel structure
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA (50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1d volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)  # Volume spike: 1.5x average
    
    # Align HTF indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20_aligned[i]
        breakout_down = close[i] < low_20_aligned[i]
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i]
        
        # Entry conditions - Donchian breakout with trend and volume
        long_entry = breakout_up and trend_up and vol_confirm
        short_entry = breakout_down and trend_down and vol_confirm
        
        # Exit conditions: return to opposite Donchian band or trend reversal
        long_exit = close[i] < low_20_aligned[i] or (trend_down and position == 1)
        short_exit = close[i] > high_20_aligned[i] or (trend_up and position == -1)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_EMA50_Volume"
timeframe = "12h"
leverage = 1.0
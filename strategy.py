#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    high_20 = np.full(len(close_1w), np.nan)
    low_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        high_20[i] = np.max(high_1w[i-20:i])
        low_20[i] = np.min(low_1w[i-20:i])
    
    # Calculate 50-period EMA on weekly (trend filter)
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Volume confirmation: above average volume
        vol_confirm = vol_ratio[i] > 1.5
        
        # Entry conditions: breakout in direction of trend with volume
        long_entry = long_breakout and above_ema and vol_confirm
        short_entry = short_breakout and below_ema and vol_confirm
        
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

name = "12h_1w_donchian_ema50_volume"
timeframe = "12h"
leverage = 1.0
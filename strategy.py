#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long: price breaks above Donchian Upper + price above 1w EMA50 + volume > 1.8x avg volume
# Short: price breaks below Donchian Lower + price below 1w EMA50 + volume > 1.8x avg volume
# Donchian levels calculated from 1d data: Upper = max(high, 20), Lower = min(low, 20)
# Trend filter: only take longs when price > EMA50, shorts when price < EMA50
# Volume confirmation reduces false breakouts
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in both bull and bear markets by using 1w EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    donch_upper = np.full(len(high_1d), np.nan)
    donch_lower = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_upper[i] = np.max(high_1d[i-20:i])
        donch_lower[i] = np.min(low_1d[i-20:i])
    
    # Align 1d Donchian to 1d timeframe (no alignment needed as we're using 1d as primary)
    # Since we're using 1d as primary timeframe, we can use the values directly
    # But we need to align to the 1d index from the prices dataframe
    # We'll handle this by using the 1d index directly in the loop
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period = 20 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    # Create a mapping from 1d index to price index for efficient lookup
    # We'll use the 1d dataframe's index to map to the prices dataframe
    # Since prices is at 1d frequency, we can use direct indexing
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_upper[i-20]) or np.isnan(donch_lower[i-20]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Get the 1w EMA value for current day
        # We need to find the corresponding 1w index for the current date
        # Since we're using 1d as primary, we'll use a simple approach:
        # Get the 1w EMA value that corresponds to the current date
        # We'll use the last available 1w EMA value
        # For simplicity in 1d timeframe, we'll use the 1w EMA from the previous week
        
        # Get 1w EMA aligned to 1d timeframe
        # We'll align the 1w EMA to 1d using the helper function
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
        
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_upper[i-20]
        lower = donch_lower[i-20]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: break above Donchian Upper + above EMA50 + volume confirmation
            if (price > upper and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian Lower + below EMA50 + volume confirmation
            elif (price < lower and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian Lower or below EMA50
            if (price < lower or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian Upper or above EMA50
            if (price > upper or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_EMA_Volume"
timeframe = "1d"
leverage = 1.0
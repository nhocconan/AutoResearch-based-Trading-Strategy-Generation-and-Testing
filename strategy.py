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
    
    # Get weekly data for HTF trend
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly Supertrend for trend direction
    # ATR calculation
    weekly_high_prev = np.concatenate([[weekly_high[0]], weekly_high[:-1]])
    weekly_low_prev = np.concatenate([[weekly_low[0]], weekly_low[:-1]])
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - weekly_close_prev)
    tr3 = np.abs(weekly_low - weekly_close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_weekly = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (weekly_high + weekly_low) / 2
    upper_band = hl2 + (3.0 * atr_weekly)
    lower_band = hl2 - (3.0 * atr_weekly)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(weekly_close)
    direction = np.ones_like(weekly_close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = weekly_close[0]
    direction[0] = 1
    
    for i in range(1, len(weekly_close)):
        # Calculate upper and lower bands
        upper_band[i] = hl2[i] + (3.0 * atr_weekly[i])
        lower_band[i] = hl2[i] - (3.0 * atr_weekly[i])
        
        # Adjust bands based on previous close
        if supertrend[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        if supertrend[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine trend direction
        if weekly_close[i] > supertrend[i-1]:
            direction[i] = 1
        elif weekly_close[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align weekly Supertrend and direction to 6h timeframe
    supertrend_6h = align_htf_to_ltf(prices, weekly, supertrend)
    direction_6h = align_htf_to_ltf(prices, weekly, direction)
    
    # Get daily data for volume and price structure
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily volume average
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA to 6h timeframe
    vol_ma_6h = align_htf_to_ltf(prices, daily, vol_ma_20)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_6h[i]) or np.isnan(direction_6h[i]) or 
            np.isnan(vol_ma_6h[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirm = volume[i] > (1.5 * vol_ma_6h[i])
        
        # Long signal: weekly uptrend + price breaks above Donchian high + volume confirmation
        if (direction_6h[i] == 1 and 
            close[i] > highest_high[i] and 
            volume_confirm):
            signals[i] = 0.25
        
        # Short signal: weekly downtrend + price breaks below Donchian low + volume confirmation
        elif (direction_6h[i] == -1 and 
              close[i] < lowest_low[i] and 
              volume_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklySupertrend_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0
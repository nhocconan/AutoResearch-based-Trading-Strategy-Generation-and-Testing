#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d high, low, close for weekly pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using weekly OHLC approximation from daily)
    # Weekly high = max of last 5 days high, weekly low = min of last 5 days low
    # weekly close = last day's close
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = close_1d  # Using daily close as weekly close approximation
    
    # Weekly pivot: P = (H+L+C)/3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - week_low
    weekly_s1 = 2 * weekly_pivot - week_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above weekly R1 with volume and above Donchian high
            if price > r1 and price > dc_high and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and below Donchian low
            elif price < s1 and price < dc_low and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below weekly pivot or Donchian low
            if price < pivot or price < dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above weekly pivot or Donchian high
            if price > pivot or price > dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
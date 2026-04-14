#!/usr/bin/env python3
"""
Hypothesis: 1-day strategy using weekly Donchian breakout with monthly volume confirmation.
Long when weekly price breaks above 20-period Donchian high with volume > 1.5x monthly average.
Short when weekly price breaks below 20-period Donchian low with volume > 1.5x monthly average.
Exit when price returns to weekly 20-period Donchian midpoint.
Designed for low turnover: ~10-20 trades/year per symbol to minimize fee drag.
Works in bull markets by catching breakouts and in bear markets by catching breakdowns.
"""

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
    
    # Load weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load monthly data for volume confirmation
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 20:
        return np.zeros(n)
    
    volume_monthly = df_monthly['volume'].values
    vol_ma_monthly = pd.Series(volume_monthly).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Weekly index (approx 4.33 weeks per month)
        idx_weekly = i // (7 * 24 * 4)  # Approximate: 7 days * 24 hours * 4 (15min bars per hour)
        if idx_weekly < 20:
            continue
        
        # Use previous weekly values to avoid look-ahead
        prev_idx_weekly = idx_weekly - 1
        if prev_idx_weekly < 0:
            continue
            
        # Get weekly Donchian values from previous week
        dh_prev = donchian_high[prev_idx_weekly] if prev_idx_weekly < len(donchian_high) else donchian_high[-1]
        dl_prev = donchian_low[prev_idx_weekly] if prev_idx_weekly < len(donchian_low) else donchian_low[-1]
        dm_prev = donchian_mid[prev_idx_weekly] if prev_idx_weekly < len(donchian_mid) else donchian_mid[-1]
        
        # Monthly index
        idx_monthly = i // (30 * 7 * 24 * 4)  # Approximate: 30 days * 7 days/week * 24 hours * 4
        if idx_monthly < 20:
            continue
        
        # Use previous monthly values to avoid look-ahead
        prev_idx_monthly = idx_monthly - 1
        if prev_idx_monthly < 0:
            continue
            
        # Get monthly volume MA from previous month
        vol_ma_prev = vol_ma_monthly[prev_idx_monthly] if prev_idx_monthly < len(vol_ma_monthly) else vol_ma_monthly[-1]
        
        # Create arrays for alignment
        dh_arr = np.full(len(df_weekly), dh_prev)
        dl_arr = np.full(len(df_weekly), dl_prev)
        dm_arr = np.full(len(df_weekly), dm_prev)
        vol_ma_arr = np.full(len(df_monthly), vol_ma_prev)
        
        dh_1d = align_htf_to_ltf(prices, df_weekly, dh_arr)[i]
        dl_1d = align_htf_to_ltf(prices, df_weekly, dl_arr)[i]
        dm_1d = align_htf_to_ltf(prices, df_weekly, dm_arr)[i]
        vol_ma_1d = align_htf_to_ltf(prices, df_monthly, vol_ma_arr)[i]
        
        if np.isnan(dh_1d) or np.isnan(dl_1d) or np.isnan(dm_1d) or np.isnan(vol_ma_1d):
            continue
        
        if position == 0:
            # Long: weekly price breaks above Donchian high with volume surge
            if close[i] > dh_1d and volume[i] > vol_ma_1d * 1.5:
                position = 1
                signals[i] = position_size
            # Short: weekly price breaks below Donchian low with volume surge
            elif close[i] < dl_1d and volume[i] > vol_ma_1d * 1.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to weekly Donchian midpoint
            if close[i] >= dm_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to weekly Donchian midpoint
            if close[i] <= dm_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_weekly_Donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0
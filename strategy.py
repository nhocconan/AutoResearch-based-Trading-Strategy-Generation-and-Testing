#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donch_high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    donch_high_20w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20w)
    donch_low_20w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20w)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in weekly Donchian
        if np.isnan(donch_high_20w_aligned[i]) or np.isnan(donch_low_20w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_price = high[i]
        low_price = low[i]
        price = close[i]
        vol = volume[i]
        
        donch_high = donch_high_20w_aligned[i]
        donch_low = donch_low_20w_aligned[i]
        vol_ma = volume_ma_20_1d_aligned[i]
        
        # Volume filter: current volume must be above daily 20-period average
        vol_filter = vol > vol_ma
        
        if position == 0:
            # Long: break above weekly Donchian high with volume
            if high_price > donch_high and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume
            elif low_price < donch_low and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below weekly Donchian low
            if price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above weekly Donchian high
            if price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0
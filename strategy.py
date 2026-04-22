#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h high/low/close for price action
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels and volume context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 6h Donchian channels (20-period)
    # Upper: highest high of last 20 periods
    # Lower: lowest low of last 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily Donchian channels (20-period) for trend filter
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily average volume for volume filter
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h timeframe
    donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_upper_1d_aligned[i]) or 
            np.isnan(donchian_lower_1d_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol_1d = avg_volume_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        volume_filter = vol > 1.5 * avg_vol_1d
        
        # Long entry: price breaks above 6h Donchian upper AND price above daily Donchian upper (uptrend)
        if position == 0 and volume_filter:
            if price > donchian_upper[i] and price > donchian_upper_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 6h Donchian lower AND price below daily Donchian lower (downtrend)
            elif price < donchian_lower[i] and price < donchian_lower_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position != 0:
            # Exit long: price breaks below 6h Donchian lower
            # Exit short: price breaks above 6h Donchian upper
            if (position == 1 and price < donchian_lower[i]) or \
               (position == -1 and price > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DonchianBreakout_DailyTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0
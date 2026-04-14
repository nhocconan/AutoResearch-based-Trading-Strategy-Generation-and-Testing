#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian breakout with daily volume confirmation
# Long when price breaks above weekly Donchian high + volume > 1.5x 20-day average
# Short when price breaks below weekly Donchian low + volume > 1.5x 20-day average
# Exit when price crosses midline of weekly Donchian channel
# Weekly trend filter provides direction, daily volume confirms breakout strength
# Designed to work in both bull and bear markets by capturing strong breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20 periods)
    donchian_len = 20
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly Donchian high and low
    donchian_high = pd.Series(high_weekly).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=donchian_len, min_periods=donchian_len).min().values
    # Weekly Donchian midline
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Load daily data ONCE for volume average
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily volume 20-period average
    vol_daily = df_daily['volume'].values
    vol_avg = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_daily, vol_avg)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donchian_len + 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg_val = vol_avg_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirmed = vol > 1.5 * vol_avg_val
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + volume confirmation
            if price > donchian_high_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below weekly Donchian low + volume confirmation
            elif price < donchian_low_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midline
            if price < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midline
            if price > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wDonchian_1dVol_Breakout_v1"
timeframe = "12h"
leverage = 1.0
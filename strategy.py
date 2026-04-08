#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v3
# Hypothesis: Weekly Donchian breakout with daily trend filter and volume confirmation.
# Enter long when price breaks above weekly Donchian high (20), price > daily EMA50, and volume > 1.5x average volume.
# Enter short when price breaks below weekly Donchian low (20), price < daily EMA50, and volume > 1.5x average volume.
# Exit when price returns to weekly Donchian midpoint or trend filter fails.
# Designed for 8-25 trades/year on 1d to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels for weekly data
    donchian_high_20 = np.full(len(high_1w), np.nan)
    donchian_low_20 = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        donchian_high_20[i] = np.max(high_1w[i-20:i])
        donchian_low_20[i] = np.min(low_1w[i-20:i])
    
    # Weekly midpoint for exit
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align weekly levels to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    donchian_mid_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(donchian_mid_20_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly midpoint or trend filter fails
            if close[i] < donchian_mid_20_aligned[i] or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly midpoint or trend filter fails
            if close[i] > donchian_mid_20_aligned[i] or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume and trend filter
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly Donchian low with volume and trend filter
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals
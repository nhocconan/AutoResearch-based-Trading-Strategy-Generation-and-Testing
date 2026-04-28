#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Enter long when price breaks above 6h Donchian upper channel, weekly pivot shows bullish bias (price above weekly CPR pivot), and volume > 2.0x 20-bar average.
# Enter short when price breaks below 6h Donchian lower channel, weekly pivot shows bearish bias (price below weekly CPR pivot), and volume > 2.0x 20-bar average.
# Exit when price reaches the opposite Donchian channel or crosses the 6h EMA34.
# Weekly CPR (Central Pivot Range) provides institutional structure; Donchian breakouts capture momentum; volume confirms strength.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and ensure robustness.

name = "6h_Donchian20_WeeklyCPR_Pivot_VolumeSpike_v1"
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
    
    # Get weekly data for CPR pivot (weekly high, low, close)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly CPR: Pivot = (H+L+C)/3, BC = (H+L)/2, TC = (Pivot - BC) + Pivot
    # We'll use the weekly pivot as the bias indicator
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 6h data for Donchian channels and EMA34
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 34:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper: highest high over 20 periods
    # Donchian lower: lowest low over 20 periods
    high_series = pd.Series(high_6h)
    low_series = pd.Series(low_6h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Calculate 6h EMA34 for exit filter
    close_6h = df_6h['close'].values
    ema_34 = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_6h, ema_34)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i]
        breakout_down = close[i] < donchian_lower_aligned[i]
        
        # Weekly CPR bias: price above/below weekly pivot
        weekly_bias_up = close[i] > weekly_pivot_aligned[i]
        weekly_bias_down = close[i] < weekly_pivot_aligned[i]
        
        # Exit conditions: price reaches opposite Donchian channel or crosses 6h EMA34
        exit_long = close[i] < donchian_lower_aligned[i] or close[i] < ema_34_aligned[i]
        exit_short = close[i] > donchian_upper_aligned[i] or close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if breakout_up and weekly_bias_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and weekly_bias_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
# 6h_Donchian_WeeklyPivot_Breakout_VolumeConfirmation
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly Donchian high (20-period) 
# with price above weekly pivot and volume above 20-period average. Enter short when price breaks 
# below weekly Donchian low with price below weekly pivot and volume confirmation.
# Uses weekly structure for trend context and Donchian breakouts for momentum.
# Targets 15-30 trades/year for low fee drag and works in both bull and bear markets.
# Weekly pivot acts as trend filter: above pivot = bullish bias, below = bearish bias.

name = "6h_Donchian_WeeklyPivot_Breakout_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        dh_val = donchian_high_aligned[i]
        dl_val = donchian_low_aligned[i]
        pivot_val = pivot_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high with price above pivot and volume confirmation
            if close[i] > dh_val and close[i] > pivot_val and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with price below pivot and volume confirmation
            elif close[i] < dl_val and close[i] < pivot_val and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low (trend reversal)
            if close[i] < dl_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high (trend reversal)
            if close[i] > dh_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
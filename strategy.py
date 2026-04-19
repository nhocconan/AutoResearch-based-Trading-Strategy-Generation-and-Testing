#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Bollinger Band squeeze breakout with volume confirmation.
# Use weekly Bollinger Band width percentile to detect squeeze (low volatility).
# When BB width is at 20th percentile or lower (squeeze), breakout of daily Donchian(20) with volume > 1.5x average.
# Exit when price crosses opposite Donchian band or BB width expands above 80th percentile.
# This captures low-volatility breakout moves that often trend, working in both bull and bear markets.
# Weekly timeframe reduces noise, daily provides timely entries.
name = "1d_WeeklyBB_Squeeze_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    def sma(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    def std_dev(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.std(arr[i-window+1:i+1])
        return result
    
    sma_20w = sma(close_1w, 20)
    std_20w = std_dev(close_1w, 20)
    upper_bb = sma_20w + 2 * std_20w
    lower_bb = sma_20w - 2 * std_20w
    bb_width = upper_bb - lower_bb
    
    # Calculate percentile rank of BB width over 50 periods
    def percentile_rank(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            if len(window_data) > 0:
                rank = (np.sum(window_data < arr[i]) / len(window_data)) * 100
                result[i] = rank
        return result
    
    bb_width_percentile = percentile_rank(bb_width, 50)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_1d, 20)
    donch_low = rolling_min(low_1d, 20)
    
    # Align indicators to daily timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Get daily average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure BB percentile (50) and Donchian (20) are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_pct = bb_width_percentile_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Squeeze condition: BB width at 20th percentile or lower (low volatility)
        is_squeeze = bb_width_pct <= 20.0
        # Expansion exit: BB width above 80th percentile
        is_expansion = bb_width_pct >= 80.0
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter on Donchian breakout during squeeze with volume
            if is_squeeze and volume_confirmed:
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price crosses below Donchian low OR BB width expansion
            if price < donch_low_val or is_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Donchian high OR BB width expansion
            if price > donch_high_val or is_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversion with Volume Confirmation and Choppiness Filter
# Long when price touches or breaks below Camarilla S3 level with volume > 1.5x average and market in ranging state (CHOP > 61.8)
# Short when price touches or breaks above Camarilla R3 level with volume > 1.5x average and market in ranging state
# Uses 1d Camarilla levels for institutional support/resistance, volume to confirm institutional interest, and Choppiness index to avoid trending markets
# Designed for 20-40 trades/year on 4h timeframe with focus on mean reversion in ranging markets

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's Camarilla levels
    ph = np.roll(high_1d, 1)  # previous high
    pl = np.roll(low_1d, 1)   # previous low
    pc = np.roll(close_1d, 1) # previous close
    
    # First day will have invalid values (rolled from last day) - handle with valid mask
    valid_day = np.arange(len(df_1d)) > 0
    
    # Calculate Camarilla levels for each day
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    
    # Only calculate for valid days (not first day)
    camarilla_s3[valid_day] = pc[valid_day] - 1.1 * (ph[valid_day] - pl[valid_day]) / 6
    camarilla_r3[valid_day] = pc[valid_day] + 1.1 * (ph[valid_day] - pl[valid_day]) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Calculate average volume (20-period) for volume confirmation
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime detection
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index"""
        atr = []
        for i in range(len(close_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], 
                         abs(high_arr[i] - close_arr[i-1]),
                         abs(low_arr[i] - close_arr[i-1]))
            atr.append(tr)
        
        atr_arr = np.array(atr)
        # Smoothed ATR (using simple mean for simplicity, can be smoothed further)
        atr_sum = pd.Series(atr_arr).rolling(window=window, min_periods=window).sum().values
        
        # Calculate highest high and lowest low over the period
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = np.minimum.reduce([np.roll(low_arr, i) for i in range(window)])  # Simplified
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.where((range_hl > 0) & (atr_sum > 0), 100 * np.log10(atr_sum / range_hl) / np.log10(window), 50)
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after indicators are ready
        # Skip if any required data is invalid
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(volume_avg[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * volume_avg[i]
        
        # Choppiness filter: market in ranging state (CHOP > 61.8)
        ranging_market = chop[i] > 61.8
        
        # Entry conditions
        long_signal = (low[i] <= camarilla_s3_aligned[i]) and volume_confirm and ranging_market
        short_signal = (high[i] >= camarilla_r3_aligned[i]) and volume_confirm and ranging_market
        
        # Exit conditions: price moves back toward the pivot level (midpoint between S3 and R3)
        pivot_level = (camarilla_s3_aligned[i] + camarilla_r3_aligned[i]) / 2
        exit_long = close[i] >= pivot_level
        exit_short = close[i] <= pivot_level
        
        # Priority: entry > exit > hold
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
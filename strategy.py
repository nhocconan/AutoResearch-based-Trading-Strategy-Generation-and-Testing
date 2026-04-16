#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with daily volume confirmation.
# Long when price breaks above weekly Donchian upper channel (20-period) AND daily volume > 1.5x 20-day average volume.
# Short when price breaks below weekly Donchian lower channel AND daily volume > 1.5x 20-day average volume.
# Exit when price crosses the weekly Donchian middle (median) line or volume drops below average.
# Uses discrete position size 0.25. Weekly Donchian provides structural trend filter to avoid whipsaws.
# Daily volume confirmation ensures breakouts have conviction. 1d timeframe targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns) with volume filter reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data once before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === Weekly Indicators: Donchian Channels (20-period) ===
    # Upper channel = highest high over 20 weeks
    # Lower channel = lowest low over 20 weeks
    # Middle channel = median of upper and lower
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # === Daily Indicators: Volume average (20-day) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1d)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_1w, middle_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < middle (trend weakening) OR volume < average (losing conviction)
            if (price < middle) or (vol < vol_ma):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > middle (trend weakening) OR volume < average (losing conviction)
            if (price > middle) or (vol < vol_ma):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > upper channel AND volume > 1.5x average (strong breakout)
            if (price > upper) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < lower channel AND volume > 1.5x average (strong breakdown)
            elif (price < lower) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wDonchian20_Breakout_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0
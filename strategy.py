#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction and 4h Donchian(20) breakout for entries.
# Enter long when price breaks above 4h Donchian upper band with volume > 2.0x average and Alligator aligned bullish (Lips > Teeth > Jaw).
# Enter short when price breaks below 4h Donchian lower band with volume > 2.0x average and Alligator aligned bearish (Lips < Teeth < Jaw).
# Exit when price touches the opposite Donchian band or Alligator alignment fails.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 80-150 total trades over 4 years.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).
# Uses 1d Alligator for slower trend filter (reduces whipsaws) and 4h Donchian for structure (proven edge on SOLUSDT).

name = "4h_WilliamsAlligator_Donchian20_Breakout_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (HTF trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: SMAs of median price
    # Median price = (high + low) / 2
    median_price = (df_1d['high'] + df_1d['low']) / 2
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 4h data for Donchian channels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian channels to 4h timeframe (no shift needed as same TF)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Alligator trend alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Exit conditions: touch opposite band or Alligator alignment fails
        long_exit = close[i] < donchian_lower_aligned[i] or not bullish_alignment
        short_exit = close[i] > donchian_upper_aligned[i] or not bearish_alignment
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_alignment
        short_entry = short_breakout and vol_confirm and bearish_alignment
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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
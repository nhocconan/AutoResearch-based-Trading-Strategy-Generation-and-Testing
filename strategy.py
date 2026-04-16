#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + CHOP > 61.8 (ranging).
# Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + CHOP > 61.8.
# Exit when price crosses Donchian midpoint or volume drops below average.
# Uses discrete position size 0.25. Donchian provides structure, volume confirms breakout strength,
# choppiness filter ensures we only trade breakouts from ranging markets (avoids false breakouts in strong trends).
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (catch upside breakouts) and bear markets (catch downside breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === Calculate Donchian Channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === 12h Indicators: Volume average ===
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # === Calculate Choppiness Index (14-period) ===
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * np.sqrt(14)
    chop_num = np.log10(atr_14.sum())
    chop = 100 * chop_num / chop_denom
    # Handle division by zero or invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    
    # Align all indicators to primary timeframe (4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14)  # Donchian and chop need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_aligned[i]
        chop_val = chop[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        mid_band = donchian_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midpoint OR volume drops below average
            if (price < mid_band) or (vol < vol_ma * 0.8):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midpoint OR volume drops below average
            if (price > mid_band) or (vol < vol_ma * 0.8):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = vol > vol_ma * 1.5
            # Choppiness filter: CHOP > 61.8 indicates ranging market (good for breakout fade/continuation)
            chop_filter = chop_val > 61.8
            
            # LONG: Price breaks above Donchian upper band + volume confirmation + chop filter
            if (price > upper_band) and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower band + volume confirmation + chop filter
            elif (price < lower_band) and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0
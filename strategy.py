#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h Donchian breakout and volume confirmation
# Long when CHOP > 61.8 (ranging) AND price breaks above Donchian(20) upper band AND volume > 1.5x average
# Short when CHOP > 61.8 (ranging) AND price breaks below Donchian(20) lower band AND volume > 1.5x average
# Exit when CHOP < 38.2 (trending) or opposite Donchian breakout occurs
# Choppiness Index identifies ranging markets ideal for mean reversion; Donchian provides breakout signals; volume confirms validity
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Choppiness Index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean()
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate Donchian channels on 12h (20-period)
    donch_high_12h = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max()
    donch_low_12h = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 12h Donchian channels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop[i]
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        
        if position == 0:
            # Long setup: ranging market (CHOP > 61.8) AND price breaks above Donchian upper band AND volume confirmation
            if chop_val > 61.8 and price > upper_band and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: ranging market (CHOP > 61.8) AND price breaks below Donchian lower band AND volume confirmation
            elif chop_val > 61.8 and price < lower_band and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trending market (CHOP < 38.2) OR price breaks below Donchian lower band
            if chop_val < 38.2 or price < lower_band:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trending market (CHOP < 38.2) OR price breaks above Donchian upper band
            if chop_val < 38.2 or price > upper_band:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Choppiness_Donchian_Volume"
timeframe = "4h"
leverage = 1.0
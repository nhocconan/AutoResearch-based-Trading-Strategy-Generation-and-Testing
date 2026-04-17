#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Donchian(20) breakout + volume confirmation + ATR-based trend filter.
Long when price breaks above 12h Donchian upper band with volume confirmation and ATR(14) > ATR(50) (strong volatility regime).
Short when price breaks below 12h Donchian lower band with volume confirmation and ATR(14) > ATR(50).
Exit when price returns to the 12h Donchian midpoint or breaks the opposite band with volume.
Uses 12h for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in low volatility regimes.
ATR ratio filter ensures we only trade when volatility is expanding, which improves breakout validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian and ATR calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) bands
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    upper_20 = high_12h_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_12h_series.rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 12h ATR(14) and ATR(50) for trend filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_12h, midpoint_20)
    atr14_aligned = align_htf_to_ltf(prices, df_12h, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_12h, atr50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for ATR50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(midpoint_20_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        # Volatility filter: ATR(14) > ATR(50) (expanding volatility)
        vol_filter = atr14_aligned[i] > atr50_aligned[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper band with volume and volatility expansion
            if (close[i] > upper_20_aligned[i] and 
                volume_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band with volume and volatility expansion
            elif (close[i] < lower_20_aligned[i] and 
                  volume_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower band with volume (reversal)
            if (close[i] <= midpoint_20_aligned[i] or 
                (close[i] < lower_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper band with volume (reversal)
            if (close[i] >= midpoint_20_aligned[i] or 
                (close[i] > upper_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian20_Breakout_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0
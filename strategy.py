#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
Long when price breaks above Donchian high with volume > 1.5x MA20 and CHOP(14) > 61.8 (range regime).
Short when price breaks below Donchian low with volume > 1.5x MA20 and CHOP(14) > 61.8.
Exit when price returns to Donchian midpoint or chop regime shifts to trending (CHOP < 38.2).
Uses 1d for CHOP regime, 4h for Donchian and volume.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CHOP regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d CHOP (Choppiness Index, 14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1] if i>0 else high[i]), abs(low[i] - close[i-1] if i>0 else low[i]))
        
        # Sum of ATR over period
        sum_atr = np.zeros_like(atr)
        for i in range(period, len(atr)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # CHOP = 100 * log10(sum_atr / (highest_high - lowest_low)) / log10(period)
        range_hl = highest_high - lowest_low
        # Avoid division by zero
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        chop = 100 * np.log10(sum_atr / range_hl) / np.log10(period)
        # Set values before period to NaN
        chop[:period-1] = np.nan
        return chop
    
    chop_14 = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
            else:
                upper[i] = np.nan
                lower[i] = np.nan
        return upper, lower
    
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    donch_mid = (donch_hi + donch_lo) / 2.0
    
    # Calculate 4h volume MA(20)
    vol_ma20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(chop_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in range regime (CHOP > 61.8)
        is_range = chop_14_aligned[i] > 61.8
        
        # Volume confirmation: volume > 1.5x MA20
        vol_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + range regime
            if close[i] > donch_hi[i] and vol_confirm and is_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + range regime
            elif close[i] < donch_lo[i] and vol_confirm and is_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if close[i] < donch_mid[i] or chop_14_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR regime shifts to trending (CHOP < 38.2)
            if close[i] > donch_mid[i] or chop_14_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ChopRange"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CI) regime filter with 1d Donchian breakout
# Long when CI > 61.8 (range) + price breaks above Donchian(20) high
# Short when CI > 61.8 (range) + price breaks below Donchian(20) low
# Exit when CI < 38.2 (trending) or price returns to Donchian midpoint
# Works in both bull and bear by focusing on range-bound markets where mean reversion works.
# Low trade frequency: only trade in ranging markets (CI > 61.8) with breakout confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d data
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align Donchian channels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Calculate 14-period Choppiness Index on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero when hh == ll
    range_hl = hh - ll
    ci_raw = np.where(range_hl != 0, tr_sum / range_hl, 1.0)
    ci = 100.0 * np.log10(ci_raw) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ci[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        ci_val = ci[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        
        # Range regime filter: CI > 61.8 indicates ranging market
        range_regime = ci_val > 61.8
        
        if position == 0:
            # Enter long when in range + price breaks above Donchian high
            if range_regime and price > donch_high_val:
                signals[i] = 0.25
                position = 1
            # Enter short when in range + price breaks below Donchian low
            elif range_regime and price < donch_low_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: trending market (CI < 38.2) or price returns to midpoint
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when market trends or price returns to Donchian midpoint
                if ci_val < 38.2 or price < donch_mid_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when market trends or price returns to Donchian midpoint
                if ci_val < 38.2 or price > donch_mid_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_CI_Donchian_Range"
timeframe = "4h"
leverage = 1.0
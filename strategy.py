#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Choppiness Index regime filter and 1d Donchian breakout.
# Uses 1d Choppiness Index to identify ranging markets (CHOP > 61.8) for mean reversion
# and trending markets (CHOP < 38.2) for trend following. In ranging markets, we mean
# revert at Donchian channel extremes; in trending markets, we follow breakouts.
# Volume confirmation ensures signal strength. Works in both bull and bear markets
# by adapting to market regime.
# Target: 75-200 total trades over 4 years (19-50/year).
name = "4h_1d_Chop_Donchian_Adaptive"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and Donchian calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # True range sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero in chop calculation
    range_hl = hh - ll
    range_hl_safe = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Choppiness Index: 100 * log10(tr_sum / range_hl) / log10(14)
    chop = 100 * np.log10(tr_sum / range_hl_safe) / np.log10(14)
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        chop_val = chop_aligned[i]
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        
        if position == 0:
            # In ranging market (CHOP > 61.8): mean reversion at Donchian extremes
            if chop_val > 61.8:
                if close[i] <= dl and volume_filter[i]:  # Near lower band -> long
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= dh and volume_filter[i]:  # Near upper band -> short
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP < 38.2): follow breakouts
            elif chop_val < 38.2:
                if close[i] > dh and volume_filter[i]:  # Break above -> long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < dl and volume_filter[i]:  # Break below -> short
                    signals[i] = -0.25
                    position = -1
            # In neutral market (38.2 <= CHOP <= 61.8): no action
            
        elif position == 1:
            # Long position management
            if chop_val > 61.8:
                # In ranging market: exit at upper band
                if close[i] >= dh:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trending/neutral: exit on breakdown
                if close[i] < dl:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short position management
            if chop_val > 61.8:
                # In ranging market: exit at lower band
                if close[i] <= dl:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trending/neutral: exit on breakout
                if close[i] > dh:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
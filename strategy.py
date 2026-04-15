#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Uses Choppiness Index (14) to filter regimes: CHOP > 61.8 = range (mean revert at Donchian bands),
# CHOP < 38.2 = trend (follow breakouts). Works in both bull and bear markets by adapting to regime.
# Volume confirmation ensures breakouts are genuine. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels and Choppiness Index
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Chop formula: 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.zeros_like(tr_sum)
    mask = range_14 > 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_14[mask]) / np.log10(14)
    
    # Align 4h indicators to higher timeframe (we're using 4h as primary, so no alignment needed for entry logic)
    # But we need to align for proper timing - wait for 4h bar close
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Regime-based logic
        chop_val = chop_aligned[i]
        
        # Long entry conditions
        long_breakout = close[i] > highest_high_aligned[i]
        vol_filter = volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1])
        
        # Short entry conditions
        short_breakout = close[i] < lowest_low_aligned[i]
        
        if chop_val > 61.8:  # Range regime - mean reversion
            # In range: sell at upper band, buy at lower band
            if long_breakout and vol_filter and position <= 0:  # Price above upper band - sell/short
                position = -1
                signals[i] = -base_size
            elif short_breakout and vol_filter and position >= 0:  # Price below lower band - buy/long
                position = 1
                signals[i] = base_size
                
        elif chop_val < 38.2:  # Trend regime - follow breakout
            # In trend: buy breakouts above, sell breakdowns below
            if long_breakout and vol_filter and position <= 0:
                position = 1
                signals[i] = base_size
            elif short_breakout and vol_filter and position >= 0:
                position = -1
                signals[i] = -base_size
        
        # Exit conditions: opposite signal or chop extreme reversal
        if position == 1 and (short_breakout or chop_val > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or chop_val < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Chop_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
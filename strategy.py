#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week Choppiness Index regime filter.
# Long: Price breaks above Donchian high(20) when weekly CHOP < 38.2 (trending) + volume > 1.5x avg volume.
# Short: Price breaks below Donchian low(20) when weekly CHOP < 38.2 + volume > 1.5x avg volume.
# Exit: Opposite Donchian break (short exit on Donchian high break, long exit on Donchian low break).
# Uses weekly Choppiness Index to filter for trending regimes only, avoiding whipsaws in ranging markets.
# Position size: 0.25 (25%) to manage drawdown during 2022 crash.
# Target: 15-30 trades per year (60-120 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index (14-period)
    chop = np.full(len(close_1w), np.nan)
    atr_1w = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        tr = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        atr_1w[i] = (atr_1w[i-1] * 13 + tr) / 14 if i > 1 else tr
    
    for i in range(14, len(close_1w)):
        atr_sum = np.sum(atr_1w[i-13:i+1])
        hh = np.max(high_1w[i-13:i+1])
        ll = np.min(low_1w[i-13:i+1])
        if hh - ll != 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # Donchian channels (20-period) on daily
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly Choppiness Index to daily (trending when CHOP < 38.2)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    trending_filter = chop_aligned < 38.2  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Donchian high break + trending regime + volume confirmation
            if (price > donch_high[i] and trending_filter[i] and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Donchian low break + trending regime + volume confirmation
            elif (price < donch_low[i] and trending_filter[i] and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Donchian low break (opposite side)
            if price < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Donchian high break (opposite side)
            if price > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Chop_Volume"
timeframe = "1d"
leverage = 1.0
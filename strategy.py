#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Donchian breakout + volume confirmation + chop regime filter.
# Uses daily Donchian channels (20-period) to identify breakouts, volume > 1.5x average for confirmation,
# and Choppiness Index < 38.2 to ensure trending market. Designed for low trade frequency
# (~15-25/year) to minimize fee decay while capturing strong momentum in both bull and bear markets.
# Works in bull markets by buying breakouts above upper band, in bear markets by selling breakdowns below lower band.

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate daily average volume (for confirmation)
    volume_1d = df_1d['volume'].values
    vol_avg_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10_1d)
    
    # Calculate Choppiness Index on daily timeframe (14-period)
    # CHOP = 100 * log10(sum(TR over n) / (max(HH,n) - min(LL,n))) / log10(n)
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop[hh - ll == 0] = 100  # Avoid division by zero
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure Donchian channels are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Chop filter: Choppiness Index < 38.2 (trending market)
        chop_filter = chop_aligned[i] < 38.2
        
        # Entry conditions: price breaks through daily Donchian levels with volume and chop confirmation
        long_entry = (high[i] > donchian_high_aligned[i] and vol_filter and chop_filter)
        short_entry = (low[i] < donchian_low_aligned[i] and vol_filter and chop_filter)
        
        # Exit conditions: price returns to opposite Donchian level
        exit_long = low[i] < donchian_low_aligned[i]
        exit_short = high[i] > donchian_high_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
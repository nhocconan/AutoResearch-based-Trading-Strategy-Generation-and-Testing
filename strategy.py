#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d regime filter (choppiness index) and volume confirmation
# - Uses 4h Donchian channel breakout for entry signals (long on upper band breakout, short on lower band breakout)
# - Uses 1d Choppiness Index (CHOP) to filter regimes: only trade when CHOP < 50 (trending market)
# - Uses 4h volume confirmation: require volume > 1.5x 20-period average volume
# - Exits on opposite Donchian band touch or when CHOP > 60 (range market)
# - Position size: 0.25 (25% of capital) to balance risk and reward
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to minimize fee drag
# - Combines price breakout with regime filter to work in both bull and bear markets
# - Works on BTC/ETH/SOL: breakouts work in trends, regime filter avoids whipsaws in ranges

name = "4h_1d_donchian_breakout_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Choppiness Index (CHOP) - measures whether market is trending or ranging
    # CHOP > 61.8 = ranging market, CHOP < 38.2 = strongly trending
    # We'll use CHOP < 50 as trending regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denominator = np.where((highest_high_14 - lowest_low_14) != 0,
                               (highest_high_14 - lowest_low_14),
                               np.nan)
    chop_raw = 100 * np.log10(atr_1d / chop_denominator * np.sqrt(14)) / np.log10(14)
    chop_1d = chop_raw  # No additional smoothing for simplicity
    
    # Align 1d Choppiness Index to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian Channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            avg_volume_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price touches lower Donchian band OR choppy regime
            if close[i] <= lowest_low_20[i]:  # Return to lower band
                position = 0
                signals[i] = 0.0
            elif chop_1d_aligned[i] > 60:  # Market becoming too choppy/ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price touches upper Donchian band OR choppy regime
            if close[i] >= highest_high_20[i]:  # Return to upper band
                position = 0
                signals[i] = 0.0
            elif chop_1d_aligned[i] > 60:  # Market becoming too choppy/ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout entries with volume confirmation and trending regime
            if (close[i] >= highest_high_20[i] and  # Break above upper band
                volume_confirmed[i] and           # Volume confirmation
                chop_1d_aligned[i] < 50):         # Trending regime (not choppy)
                position = 1
                signals[i] = 0.25
            elif (close[i] <= lowest_low_20[i] and   # Break below lower band
                  volume_confirmed[i] and            # Volume confirmation
                  chop_1d_aligned[i] < 50):          # Trending regime (not choppy)
                position = -1
                signals[i] = -0.25
    
    return signals
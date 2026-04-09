#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d choppiness regime filter
# - Primary signal: Price breaks above/below 12h Donchian channel (20-period high/low)
# - Volume confirmation: 1d volume > 1.5 * 20-period median volume (ensures participation)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) for mean reversion at channel extremes
# - Position size: 0.25 (discrete level) to balance return and fee drag
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, chop filter avoids false breakouts in ranges

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = volume_1d > (1.5 * median_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1d Choppiness Index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of True Range over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    hl_range = highest_high_14 - lowest_low_14
    chop_raw = np.where(hl_range != 0,
                        100 * np.log10(atr_14 / hl_range) / np.log10(14),
                        50)  # neutral when no range
    chop_below = chop_raw < 61.8  # True when trending (chop < 61.8)
    chop_below_aligned = align_htf_to_ltf(prices, df_1d, chop_below)
    
    # Pre-compute 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_below_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below Donchian low OR chop regime shifts to trending (loss of range)
            if close_12h[i] < donchian_low[i] or chop_below_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above Donchian high OR chop regime shifts to trending
            if close_12h[i] > donchian_high[i] or chop_below_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and chop regime (range)
            # Long: Price breaks above Donchian high AND volume spike AND chop > 61.8 (range)
            if (close_12h[i] > donchian_high[i] and 
                volume_spike_aligned[i] and 
                not chop_below_aligned[i]):  # chop > 61.8 when not chop_below
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below Donchian low AND volume spike AND chop > 61.8 (range)
            elif (close_12h[i] < donchian_low[i] and 
                  volume_spike_aligned[i] and 
                  not chop_below_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
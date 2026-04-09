#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and chop regime filter
# - Primary signal: 12h price breaks above/below Donchian channel (20-period high/low)
# - Volume confirmation: 1d volume > 1.5x 20-period average volume (avoid low-participation breakouts)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) for mean reversion at Donchian bounds
# - Exit: Price returns to Donchian midpoint (mean reversion) or opposite breakout
# - Position size: 0.25 (discrete level) to minimize fee churn
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
    
    # Pre-compute 1d average volume for volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    # Pre-compute 1d high, low, close for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1))).values
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop = np.where(chop_denominator > 0,
                    100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14),
                    50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_volume_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        volume_spike = volume_1d[i] > 1.5 * avg_volume_20_aligned[i] if not np.isnan(volume_1d[i]) else False
        
        # Chop regime: > 61.8 = ranging market (good for mean reversion at Donchian bounds)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint OR breaks below Donchian low
            if close_12h[i] <= donchian_mid[i] or close_12h[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint OR breaks above Donchian high
            if close_12h[i] >= donchian_mid[i] or close_12h[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and chop regime
            # Long: price breaks above Donchian high WITH volume spike AND chop regime (range)
            if (close_12h[i] > donchian_high[i] and 
                volume_spike and 
                chop_regime):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low WITH volume spike AND chop regime (range)
            elif (close_12h[i] < donchian_low[i] and 
                  volume_spike and 
                  chop_regime):
                position = -1
                signals[i] = -0.25
    
    return signals
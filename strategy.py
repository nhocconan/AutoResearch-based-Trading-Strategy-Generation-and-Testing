#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# - Choppiness Index > 61.8 = ranging market (mean revert at Donchian bands)
# - Choppiness Index < 38.2 = trending market (breakout follow)
# - Long: price breaks above Donchian(20) high + volume > 1.5x average volume + Choppiness < 38.2
# - Short: price breaks below Donchian(20) low + volume > 1.5x average volume + Choppiness < 38.2
# - Exit: opposite Donchian breakout or Choppiness > 61.8 (range) for mean reversion exit
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for Donchian calculation (same timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR) for Choppiness Index
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR (14-period) for Choppiness Index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    # Calculate Choppiness Index
    chop = 100 * np.log10(atr_14 * 14 / range_14) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(chop_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Choppiness regime: < 38.2 = trending, > 61.8 = ranging
        chop_trending = chop_4h[i] < 38.2
        chop_ranging = chop_4h[i] > 61.8
        
        if position == 0:
            # Long entry: breakout above Donchian high + volume + trending regime
            if close[i] > donchian_high[i] and volume_confirm and chop_trending:
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below Donchian low + volume + trending regime
            elif close[i] < donchian_low[i] and volume_confirm and chop_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low OR ranging market (mean reversion)
            if close[i] < donchian_low[i] or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high OR ranging market (mean reversion)
            if close[i] > donchian_high[i] or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chop_Donchian_Volume_Breakout"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# Choppiness Index > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
# In trending regime: Donchian breakout (20-period) with volume spike
# In ranging regime: Mean reversion at Donchian bands (buy lower band, sell upper band)
# Uses 1d Donchian for structure, 4h Choppiness for regime, volume for confirmation
# Designed to work in both bull (trend following) and bear (mean reversion in ranges)
# Target: 20-35 trades/year to avoid fee drag
name = "4h_Chop_Donchian1d_Regime_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 4h Choppiness Index (14-period) for regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True Range sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # 4h ATR for stops
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(chop[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        # Regime detection
        is_trending = chop[i] < 38.2  # Trending market
        is_ranging = chop[i] > 61.8   # Ranging market
        
        if position == 0:
            if is_trending:
                # Trending regime: Donchian breakout with volume
                if close[i] > donchian_high_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: Mean reversion at Donchian bands
                if close[i] <= donchian_low_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_high_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if is_trending:
                # In trending regime: exit on Donchian reversal or 2x ATR stop
                if close[i] < donchian_low_aligned[i] or close[i] < close[i-1] - 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime: exit at opposite Donchian band or 2x ATR stop
                if close[i] >= donchian_high_aligned[i] or close[i] < close[i-1] - 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if is_trending:
                # In trending regime: exit on Donchian reversal or 2x ATR stop
                if close[i] > donchian_high_aligned[i] or close[i] > close[i-1] + 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime: exit at opposite Donchian band or 2x ATR stop
                if close[i] <= donchian_low_aligned[i] or close[i] > close[i-1] + 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
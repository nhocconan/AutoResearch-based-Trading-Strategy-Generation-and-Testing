#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + 1d Donchian breakout with volume confirmation
# Uses Choppiness Index to identify ranging markets (CHOP > 61.8) for mean reversion
# Trades Donchian breakouts only in ranging markets to avoid false breakouts in trends
# Uses volume confirmation to filter weak breakouts
# Works in both bull and bear markets by trading mean reversion in ranges
# Target: 20-30 trades/year to avoid fee drag
name = "12h_Choppiness_Donchian_MeanRev_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 12h Choppiness Index (14-period)
    tr12 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr12[0] = high[0] - low[0]
    atr12 = pd.Series(tr12).rolling(window=14, min_periods=14).mean().values
    
    # True range sum over 14 periods
    tr_sum = pd.Series(tr12).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) and min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (max_high - min_low)) / log10(14)
    # Avoid division by zero
    range_14 = max_high - min_low
    chop = np.full_like(close, 50.0)  # default to neutral
    mask = (range_14 > 0) & (~np.isnan(tr_sum))
    chop[mask] = 100 * np.log10(tr_sum[mask] / range_14[mask]) / np.log10(14)
    
    # 12h ATR for position sizing
    atr_12h = pd.Series(tr12).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(chop[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        # Choppiness filter: ranging market (CHOP > 61.8)
        ranging_market = chop[i] > 61.8
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + ranging market
            if price > donchian_high_aligned[i] and volume_filter and ranging_market:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + ranging market
            elif price < donchian_low_aligned[i] and volume_filter and ranging_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters Donchian channel or chop drops below 38.2 (trending)
            if price < donchian_low_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters Donchian channel or chop drops below 38.2 (trending)
            if price > donchian_high_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4-period ATR for Choppiness Index
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr4 = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
    
    # Calculate True Range sum over 14 periods for Choppiness denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max(high) and min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (max_high - min_low)) / log10(14)
    # Avoid division by zero and invalid values
    range_hl = max_high - min_low
    chop_raw = np.where((range_hl > 0) & (~np.isnan(tr_sum)), tr_sum / range_hl, np.nan)
    chop = np.where(~np.isnan(chop_raw), 100 * np.log10(chop_raw) / np.log10(14), np.nan)
    
    # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower bands (20-period high/low)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_6h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need chop, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_align[i]) or 
            np.isnan(donch_high_6h[i]) or 
            np.isnan(donch_low_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_align[i] > 61.8
        
        if position == 0:
            # Long: price touches or breaks above Donchian upper band in ranging market with volume
            if (ranging_market and volume_filter and close[i] >= donch_high_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below Donchian lower band in ranging market with volume
            elif (ranging_market and volume_filter and close[i] <= donch_low_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below Donchian lower band (mean reversion)
            if close[i] <= donch_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above Donchian upper band (mean reversion)
            if close[i] >= donch_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Chop_Donchian_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0
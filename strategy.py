#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d choppiness regime filter
# - Entry: Long when price breaks above 4h Donchian upper channel (20) + 1h volume > 1.5x 20-period average + 1d Choppiness Index > 61.8 (range regime)
#          Short when price breaks below 4h Donchian lower channel (20) + 1h volume > 1.5x 20-period average + 1d Choppiness Index > 61.8
# - Exit: Close-based reversal - exit long when price < 4h Donchian middle (20), exit short when price > 4h Donchian middle (20)
# - Position sizing: 0.20 (discrete levels to minimize fee churn)
# - Uses 4h Donchian channels for structure, 1h volume spike for confirmation of participation,
#   and 1d Choppiness Index to filter for range-bound markets where mean reversion works best
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within HARD MAX: 200 total
# - Choppiness Index > 61.8 indicates ranging market (ideal for mean reversion), < 38.2 indicates trending
# - Volume spike ensures genuine participation at breakout, reducing false signals
# - Uses 4h for signal direction (per rules) and 1h only for entry timing precision

name = "1h_4h_1d_donchian_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Pre-compute 4h data for Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d data for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Calculate 1h volume moving average (20-period)
    volume_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over period
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(tr) / (hh - ll)) / log10(period)
    # Avoid division by zero
    range_hl = hh_1d - ll_1d
    choppiness = np.zeros_like(sum_tr)
    mask = (range_hl > 0) & (~np.isnan(sum_tr)) & (~np.isnan(range_hl))
    choppiness[mask] = 100 * np.log10(sum_tr[mask] / range_hl[mask]) / np.log10(14)
    
    # Align all HTF data to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_1h)  # Using 4h df for 1h volume MA alignment
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(choppiness_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1h close and volume
        close_price = close_1h[i]
        volume_price = volume_1h[i]
        
        # Volume confirmation: > 1.5x 20-period average
        volume_confirmation = volume_price > 1.5 * volume_ma_aligned[i]
        
        # Choppiness filter: > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = choppiness_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian upper channel + volume confirmation + ranging market
            if (close_price > upper_4h_aligned[i] and 
                volume_confirmation and 
                chop_filter):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 4h Donchian lower channel + volume confirmation + ranging market
            elif (close_price < lower_4h_aligned[i] and 
                  volume_confirmation and 
                  chop_filter):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < 4h Donchian middle channel
            # Exit short when price > 4h Donchian middle channel
            if position == 1:
                if close_price < middle_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close_price > middle_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals
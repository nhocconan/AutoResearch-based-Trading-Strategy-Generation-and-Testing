#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Spike and Choppiness Filter
# Uses Donchian(20) breakouts as primary signal with volume confirmation and choppiness regime filter
# Works in bull markets by capturing breakouts and in bear markets by filtering false breakouts during chop
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max and min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_close - min_close
    chop = np.full_like(close_1d, 50.0)  # Default to neutral
    valid_range = range_14 != 0
    chop[valid_range] = 100 * np.log10(atr_14[valid_range] * np.sqrt(14) / range_14[valid_range]) / np.log10(14)
    
    # Choppiness filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_filter = chop < 61.8  # Allow trending and neutral, avoid strong chop
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume and chop filter
            if price > highest_high[i] and vol_spike_aligned[i] > 0.5 and chop_filter_aligned[i] > 0.5:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower band with volume and chop filter
            elif price < lowest_low[i] and vol_spike_aligned[i] > 0.5 and chop_filter_aligned[i] > 0.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches Donchian lower band or conditions deteriorate
            if price < lowest_low[i] or vol_spike_aligned[i] < 0.5 or chop_filter_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches Donchian upper band or conditions deteriorate
            if price > highest_high[i] or vol_spike_aligned[i] < 0.5 or chop_filter_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dVolume_Chop"
timeframe = "4h"
leverage = 1.0
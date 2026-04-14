#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and Daily Chop Filter
# Uses 4-hour Donchian channel breakouts (20-period) for trend following
# Confirmed by volume spike (>1.5x 20-period average) and daily chop regime (<61.8 for trending)
# Works in both bull and bear markets by capturing breakouts with volume confirmation
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily choppy market index (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = high_1d[0] - low_1d[0]
    
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chop value: 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    # Using 20-period for more stability
    sum_tr = pd.Series(tr).rolling(window=20, min_periods=20).sum().values
    max_hh = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    min_ll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(sum_tr / (max_hh - min_ll)) / np.log10(20)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # avoid division by zero
    chop_1d = chop
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is trending (chop < 61.8)
        is_trending = chop_1d_aligned[i] < 61.8
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume confirmation
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                is_trending):
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below Donchian low with volume confirmation
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  is_trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retouches Donchian low or chop increases too much
            if close[i] <= lowest_low[i] or chop_1d_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price retouches Donchian high or chop increases too much
            if close[i] >= highest_high[i] or chop_1d_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0
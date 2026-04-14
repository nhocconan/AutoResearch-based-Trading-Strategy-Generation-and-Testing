#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 12-hour Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Long when: price > Donchian upper (20) AND CHOP(12h) > 61.8 (range) AND volume > 1.5x avg
# Short when: price < Donchian lower (20) AND CHOP(12h) > 61.8 (range) AND volume > 1.5x avg
# Exit when price crosses back inside Donchian channel
# In ranging markets (high chop), mean reversion at channel extremes works well.
# In trending markets (low chop), we avoid false breakouts by requiring high chop.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index on 12h (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with same length
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max(high, close_prev) and min(low, close_prev) over 14 periods
    high_close_prev = np.maximum(high_12h, np.concatenate([[np.nan], close_12h[:-1]]))
    low_close_prev = np.minimum(low_12h, np.concatenate([[np.nan], close_12h[:-1]]))
    
    max_hc = pd.Series(high_close_prev).rolling(window=14, min_periods=14).max().values
    min_lc = pd.Series(low_close_prev).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_hc - min_lc
    range_14 = np.where(range_14 == 0, np.nan, range_14)
    
    # Choppiness Index = 100 * log10(sum(TR14) / (range14)) / log10(14)
    chop_raw = 100 * np.log10(atr_14 / range_14) / np.log10(14)
    chop_12h = chop_raw  # already aligned to 12h index
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        chop_value = chop_aligned[i]
        
        # Only trade in ranging markets (high chop > 61.8)
        if chop_value > 61.8:
            if position == 0:
                # Long setup: breakout above Donchian high + volume confirmation
                if price > high_20[i] and vol > vol_threshold:
                    position = 1
                    signals[i] = position_size
                # Short setup: breakdown below Donchian low + volume confirmation
                elif price < low_20[i] and vol > vol_threshold:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long: price falls back below Donchian low (opposite band)
                if price < low_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: price rises back above Donchian high (opposite band)
                if price > high_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            # In trending markets (low chop), stay flat to avoid false breakouts
            signals[i] = 0.0
            position = 0  # force flat when not ranging
    
    return signals

name = "4h_12hChop_Donchian_Volume"
timeframe = "4h"
leverage = 1.0
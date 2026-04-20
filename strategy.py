#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# Choppiness > 61.8 indicates range (mean revert at Donchian boundaries), < 38.2 indicates trend (follow breakout)
# This regime filter reduces false breakouts in sideways markets, improving performance in both bull and bear markets
# Target: 20-50 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channels
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-period ATR for Choppiness Index
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True Range for Choppiness calculation (same as ATR calculation)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate 14-period max(high) - min(low)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index: 100 * log10(tr_sum / range_max_min) / log10(14)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.where((range_max_min == 0) | (tr_sum <= 0), 50, chop)  # neutral when undefined
    
    # Align all 1d indicators and chop to 4h timeframe
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if (np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_value = chop_aligned[i]
        
        # Regime filter: only trade in trending markets (Choppiness < 38.2) or strong range (Choppiness > 61.8)
        # In trending regime: follow breakouts
        # In ranging regime: mean revert at Donchian boundaries
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        resistance = highest_20d_aligned[i]
        support = lowest_20d_aligned[i]
        price = close[i]
        
        if position == 0:
            # In trending regime: follow breakouts
            if is_trending and vol_filter:
                # Long: price breaks above 20-day resistance
                if price > resistance:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price breaks below 20-day support
                elif price < support:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            # In ranging regime: mean revert at boundaries
            elif is_ranging and vol_filter:
                # Long: price near support (within 0.5% of support)
                if price <= support * 1.005:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price near resistance (within 0.5% of resistance)
                elif price >= resistance * 0.995:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (1.5x ATR below entry) or opposite signal
            if is_trending:
                # In trending: exit on breakdown of support
                if price < support:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging: exit when price moves to middle or resistance
                if price >= resistance * 0.995 or price <= (resistance + support) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (1.5x ATR above entry) or opposite signal
            if is_trending:
                # In trending: exit on breakout of resistance
                if price > resistance:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging: exit when price moves to middle or support
                if price <= support * 1.005 or price >= (resistance + support) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Donchian_MeanRev_Trend"
timeframe = "4h"
leverage = 1.0
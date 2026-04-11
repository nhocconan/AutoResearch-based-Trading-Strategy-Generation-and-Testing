#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# - Donchian breakout captures strong directional moves
# - Volume confirmation ensures institutional participation
# - Choppiness regime filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, < 38.2 = trend)
# - Works in both bull and bear markets by following established trends
# - Target: 20-40 trades/year to minimize fee drag while capturing significant moves

name = "4h_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for choppiness calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, tr_sum_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50.0  # Default to neutral when invalid
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume SMA for confirmation (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Choppiness regime filter: only trade when trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        # Donchian breakout signals
        breakout_long = close[i] > highest_high_20[i-1]  # Break above previous period's high
        breakout_short = close[i] < lowest_low_20[i-1]   # Break below previous period's low
        
        # Entry conditions
        enter_long = breakout_long and vol_confirm and is_trending
        enter_short = breakout_short and vol_confirm and is_trending
        
        # Exit conditions: reverse signal or volatility expansion
        exit_long = breakout_short or (chop_aligned[i] > 61.8)  # Exit on short breakout or ranging market
        exit_short = breakout_long or (chop_aligned[i] > 61.8)  # Exit on long breakout or ranging market
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
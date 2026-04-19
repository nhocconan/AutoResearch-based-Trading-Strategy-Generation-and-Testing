#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + Choppiness regime filter
# - Long when price breaks above Donchian(20) high with volume > 1.5x average AND Choppiness > 61.8 (range)
# - Short when price breaks below Donchian(20) low with volume > 1.5x average AND Choppiness > 61.8 (range)
# - Exit when price crosses Donchian midpoint (10-period average of high/low) or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to capture breakouts in ranging markets (chop regime) while avoiding trending markets
# - Target: 20-40 trades/year to minimize fee drag

name = "4h_Donchian20_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and Choppiness
    df_4h = get_htf_data(prices, '4h')
    
    # Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Choppiness Index (14-period)
    atr_4h = pd.Series(np.maximum(np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(high_4h, 1))), np.abs(low_4h - np.roll(low_4h, 1)))).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    min_ll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(sum_atr / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # avoid division by zero
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h by dividing by 6 (6x 4h bars in 1d)
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6)
        
        # Chop filter: range-bound market (Choppiness > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for long entry: break above Donchian high + volume + chop
            if close[i] > donchian_high_aligned[i] and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: break below Donchian low + volume + chop
            elif close[i] < donchian_low_aligned[i] and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses Donchian mid or breakdown
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses Donchian mid or breakout
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
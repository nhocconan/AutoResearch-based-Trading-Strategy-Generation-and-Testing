#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Bollinger Bands width filter and 1w Donchian breakout
# Bollinger Bands width identifies low volatility (squeeze) conditions
# When volatility is low, breakouts from Donchian channels (1w) tend to be strong
# Works in both bull and bear markets as it captures volatility expansion phases
# Uses 1d BB width for regime filter and 1w Donchian for direction - avoids overtrading

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    
    # Basis (SMA)
    basis = pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Deviation
    dev = bb_mult * pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and Lower bands
    upper = basis + dev
    lower = basis - dev
    # Bandwidth (normalized by basis to make it scale-invariant)
    bb_width = (upper - lower) / basis
    bb_width = np.where(basis == 0, 0, bb_width)  # Avoid division by zero
    
    # Align BB width to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Load 1w data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian channels (20 periods)
    donch_length = 20
    donch_high = pd.Series(df_1w['high']).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Align Donchian channels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 20)  # Need enough for BB and Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_width_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Regime filter: Low volatility (BB width below 20th percentile of last 50 periods)
        # We calculate percentile rank manually to avoid look-ahead
        if i >= 50:
            bb_width_slice = bb_width_aligned[max(0, i-50):i]
            if len(bb_width_slice) > 0:
                # Calculate percentile of current BB width vs last 50 values
                sorted_widths = np.sort(bb_width_slice[~np.isnan(bb_width_slice)])
                if len(sorted_widths) > 0:
                    current_width = bb_width_aligned[i]
                    # Count how many values are less than current width
                    rank = np.searchsorted(sorted_widths, current_width, side='left')
                    percentile = (rank / len(sorted_widths)) * 100
                    low_vol = percentile <= 20  # Low volatility regime
                else:
                    low_vol = False
            else:
                low_vol = False
        else:
            low_vol = False
        
        # Breakout signals from 1w Donchian
        breakout_up = price > donch_high_aligned[i]
        breakout_down = price < donch_low_aligned[i]
        
        if position == 0:
            # Enter long: low volatility + upward breakout
            if low_vol and breakout_up:
                position = 1
                signals[i] = position_size
            # Enter short: low volatility + downward breakout
            elif low_vol and breakout_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR high volatility (BB width > 80th percentile)
            if i >= 50:
                bb_width_slice = bb_width_aligned[max(0, i-50):i]
                if len(bb_width_slice) > 0:
                    sorted_widths = np.sort(bb_width_slice[~np.isnan(bb_width_slice)])
                    if len(sorted_widths) > 0:
                        current_width = bb_width_aligned[i]
                        rank = np.searchsorted(sorted_widths, current_width, side='left')
                        percentile = (rank / len(sorted_widths)) * 100
                        high_vol = percentile >= 80
                    else:
                        high_vol = False
                else:
                    high_vol = False
            else:
                high_vol = False
            
            if price < donch_low_aligned[i] or high_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR high volatility
            if i >= 50:
                bb_width_slice = bb_width_aligned[max(0, i-50):i]
                if len(bb_width_slice) > 0:
                    sorted_widths = np.sort(bb_width_slice[~np.isnan(bb_width_slice)])
                    if len(sorted_widths) > 0:
                        current_width = bb_width_aligned[i]
                        rank = np.searchsorted(sorted_widths, current_width, side='left')
                        percentile = (rank / len(sorted_widths)) * 100
                        high_vol = percentile >= 80
                    else:
                        high_vol = False
                else:
                    high_vol = False
            else:
                high_vol = False
            
            if price > donch_high_aligned[i] or high_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dBBwidth_1wDonchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0
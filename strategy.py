#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ATR-based volatility filter and 1w Donchian breakout
# The strategy combines long-term trend direction (1w Donchian) with volatility regime filtering (1d ATR).
# Low volatility periods (ATR below 20-day percentile) precede breakouts, which are then filtered by 1w Donchian breakout direction.
# This approach aims to capture volatility expansion after contraction, working in both bull and bear markets.
# Uses discrete position sizing (0.25) to minimize trade frequency and fee impact.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR (14-period)
    atr_length = 14
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=atr_length, min_periods=atr_length).mean().values
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Load 1w data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian channels (20 periods)
    donch_length = 20
    donch_high = pd.Series(df_1w['high']).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Align Donchian channels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need enough for ATR and Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: Low volatility (ATR below 20th percentile of last 30 periods)
        # Calculate percentile rank manually to avoid look-ahead
        if i >= 30:
            atr_slice = atr_aligned[max(0, i-30):i]
            if len(atr_slice) > 0:
                # Remove NaN values
                clean_atr = atr_slice[~np.isnan(atr_slice)]
                if len(clean_atr) > 0:
                    sorted_atr = np.sort(clean_atr)
                    current_atr = atr_aligned[i]
                    # Count how many values are less than current ATR
                    rank = np.searchsorted(sorted_atr, current_atr, side='left')
                    percentile = (rank / len(sorted_atr)) * 100
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
            # Exit long: price breaks below Donchian low OR high volatility (ATR > 80th percentile)
            if i >= 30:
                atr_slice = atr_aligned[max(0, i-30):i]
                if len(atr_slice) > 0:
                    clean_atr = atr_slice[~np.isnan(atr_slice)]
                    if len(clean_atr) > 0:
                        sorted_atr = np.sort(clean_atr)
                        current_atr = atr_aligned[i]
                        rank = np.searchsorted(sorted_atr, current_atr, side='left')
                        percentile = (rank / len(sorted_atr)) * 100
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
            if i >= 30:
                atr_slice = atr_aligned[max(0, i-30):i]
                if len(atr_slice) > 0:
                    clean_atr = atr_slice[~np.isnan(atr_slice)]
                    if len(clean_atr) > 0:
                        sorted_atr = np.sort(clean_atr)
                        current_atr = atr_aligned[i]
                        rank = np.searchsorted(sorted_atr, current_atr, side='left')
                        percentile = (rank / len(sorted_atr)) * 100
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

name = "12h_1dATR_1wDonchian_VolatilityBreakout_v1"
timeframe = "12h"
leverage = 1.0
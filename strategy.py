#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Williams %R for mean reversion and 1d volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extremes tend to work in both bull and bear markets
# Volume confirmation ensures moves have conviction; combined with mean reversion at extremes reduces false signals
# Uses 1d Williams %R (14 period) and 1d volume SMA (20 period) for regime filter
# Designed to generate ~20-40 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 period)
    williams_length = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=williams_length, min_periods=williams_length).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_length, min_periods=williams_length).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 1d volume SMA (20 period) for volume confirmation
    vol_sma_length = 20
    vol_1d = df_1d['volume'].values
    vol_sma = pd.Series(vol_1d).rolling(window=vol_sma_length, min_periods=vol_sma_length).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_sma_aligned = align_htf_to_ltf(prices, df_1d, vol_sma)
    
    # Current volume (use current bar's volume for comparison)
    # Note: We compare current 4h volume against the aligned 1d volume SMA
    # This requires estimating how much volume corresponds to the current 4h bar
    # Since we don't have intraday volume breakdown, we use the current bar's volume directly
    # and compare it to the 1d average scaled appropriately (1d has ~26 4h bars in 24h)
    # Simplified: use current volume > 1.5 * (1d vol_sma / 26) as proxy for above-average volume
    vol_threshold = vol_sma_aligned / 26.0 * 1.5  # Approximate threshold for significant volume
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, williams_length, vol_sma_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_sma_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        williams_r_val = williams_r_aligned[i]
        vol_ratio = volume[i] / vol_threshold[i] if vol_threshold[i] > 0 else 0
        
        # Mean reversion signals from Williams %R extremes
        # Oversold: Williams %R < -80 (potential long)
        # Overbought: Williams %R > -20 (potential short)
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        high_volume = vol_ratio > 1.0  # Volume above threshold
        
        if position == 0:
            # Enter long: oversold + high volume
            if oversold and high_volume:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + high volume
            elif overbought and high_volume:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (middle) OR low volume
            exit_signal = williams_r_val > -50 or vol_ratio < 0.5
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (middle) OR low volume
            exit_signal = williams_r_val < -50 or vol_ratio < 0.5
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dWilliamsR_Volume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0
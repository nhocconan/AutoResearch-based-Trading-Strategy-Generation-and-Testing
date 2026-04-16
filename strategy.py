#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R mean reversion with volume spike confirmation.
# Long when 1d Williams %R < -80 (oversold) + 6h volume > 1.5x 20-period median volume.
# Short when 1d Williams %R > -20 (overbought) + 6h volume > 1.5x 20-period median volume.
# Uses discrete position size 0.25. Exits when Williams %R returns to -50 level.
# Williams %R identifies extreme price levels that often reverse in crypto markets.
# Volume confirmation ensures institutional participation at extremes.
# 6h timeframe targets 12-37 trades/year to minimize fee drag. Works in both bull and bear markets by fading extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14))
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # === 1d Indicators: Williams %R signal levels ===
    williams_r_oversold = -80  # Long signal
    williams_r_overbought = -20  # Short signal
    williams_r_exit = -50  # Exit level
    
    # Align Williams %R to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 6h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20)  # Williams %R needs 14, volume median needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        vol_median = vol_median_20[i]
        
        # Volume spike filter: current 6h volume > 1.5x median volume
        volume_spike = volume[i] > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R returns to -50 level
            if williams_r_val >= williams_r_exit:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to -50 level
            if williams_r_val <= williams_r_exit:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + volume spike
            if (williams_r_val < williams_r_oversold) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) + volume spike
            elif (williams_r_val > williams_r_overbought) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dWilliamsR_MeanReversion_VolumeSpike1.5x_V1"
timeframe = "6h"
leverage = 1.0
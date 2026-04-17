#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_Tight
4-hour strategy using Camarilla pivot levels (R1/S1) with volume spike confirmation.
Enters long when price breaks above R1 with volume spike >2x average.
Enters short when price breaks below S1 with volume spike >2x average.
Uses tight volume confirmation to limit trades and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Points (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R1 = close_1d + (range_1d * 1.0833)
    S1 = close_1d - (range_1d * 1.0833)
    
    # Align to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === Volume Spike Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get current day's volume for spike detection
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(vol_1d_current[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current day's volume > 2x 20-day average
        vol_spike = vol_1d_current[i] > 2.0 * vol_ma_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        # Exit conditions: return to opposite pivot level
        exit_long = close[i] < S1_aligned[i]
        exit_short = close[i] > R1_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume spike
            if breakout_long and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume spike
            elif breakout_short and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_Tight"
timeframe = "4h"
leverage = 1.0
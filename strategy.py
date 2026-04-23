#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d Williams %R filter and volume confirmation
- Uses Camarilla pivot levels from 4h timeframe for institutional support/resistance
- 1d Williams %R(14) defines higher timeframe momentum: only trade breakouts when momentum is not extreme
- Volume confirmation (> 1.8x 20-period average) filters false breakouts while reducing trade frequency
- Designed for 4h timeframe targeting 25-40 trades/year (100-160 over 4 years)
- Williams %R helps avoid entering during overextended conditions, improving win rate in ranging markets
- Camarilla R3/S3 levels provide stronger breakout signals than R1/S1 with fewer whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Calculate 1d Williams %R(14) for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: > 1.8x 20-period average (balanced to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 with Williams %R not oversold (> -80) and volume spike
            long_breakout = (close[i] > camarilla_r3_aligned[i] and 
                           williams_r_aligned[i] > -80 and
                           volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below Camarilla S3 with Williams %R not overbought (< -20) and volume spike
            short_breakout = (close[i] < camarilla_s3_aligned[i] and 
                            williams_r_aligned[i] < -20 and
                            volume[i] > 1.8 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or Williams %R extreme reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 or Williams %R becomes oversold (< -80)
                if (close[i] < camarilla_s3_aligned[i] or 
                    williams_r_aligned[i] < -80):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R3 or Williams %R becomes overbought (> -20)
                if (close[i] > camarilla_r3_aligned[i] or 
                    williams_r_aligned[i] > -20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dWilliamsR_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
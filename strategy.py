#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 12h EMA Trend and Volume Confirmation
- Uses 12h EMA(50) for trend filter to ensure alignment with higher timeframe
- Camarilla pivot levels (R1,S1,R2,S2) calculated from 1d OHLC for structure
- Breakout above R1 or below S1 with volume confirmation (>1.5x 20-bar average)
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading breakouts in direction of 12h trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.25*(high-low)*1.1/2, etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using close of previous day)
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    R2 = close_1d + range_1d * 1.1 / 6
    S2 = close_1d - range_1d * 1.1 / 6
    R3 = close_1d + range_1d * 1.1 / 4
    S3 = close_1d - range_1d * 1.1 / 4
    R4 = close_1d + range_1d * 1.1 / 2
    S4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: price breaks above R1 (first resistance) + uptrend + volume spike
        # Short: price breaks below S1 (first support) + downtrend + volume spike
        long_signal = (close[i] > R1_aligned[i] and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < S1_aligned[i] and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below S1 (first support)
                if (close[i] < ema_50_12h_aligned[i] or 
                    close[i] < S1_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above R1 (first resistance)
                if (close[i] > ema_50_12h_aligned[i] or 
                    close[i] > R1_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
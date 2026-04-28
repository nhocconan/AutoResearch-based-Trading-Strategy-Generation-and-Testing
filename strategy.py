#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and 4h Supertrend trend filter.
# Enter long when price breaks above R3 with volume spike and Supertrend uptrend.
# Enter short when price breaks below S3 with volume spike and Supertrend downtrend.
# Uses discrete position sizing (0.30) to minimize fee churn. Target: 75-150 total trades over 4 years.
# Designed to work in both bull and bear markets via regime-adaptive Supertrend filter.

name = "4h_Camarilla_R3S3_4hSupertrend_Trend_VolumeSpike_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    R3 = np.full(n_1d, np.nan)
    S3 = np.full(n_1d, np.nan)
    PP = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        # Camarilla pivot calculation
        PP[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        range_1d = high_1d[i] - low_1d[i]
        R3[i] = PP[i] + range_1d * 1.1 / 4.0
        S3[i] = PP[i] - range_1d * 1.1 / 4.0
    
    # Forward fill to get most recent pivot levels
    R3 = pd.Series(R3).ffill().values
    S3 = pd.Series(S3).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 1d Camarilla levels to 4h timeframe with 1-bar delay for confirmation
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Get 4h data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2.0
    upperband = hl2 + 3.0 * atr_10
    lowerband = hl2 - 3.0 * atr_10
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = 0
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_4h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Forward fill Supertrend values
    supertrend = pd.Series(supertrend).ffill().values
    direction = pd.Series(direction).ffill().values
    
    # Align Supertrend to 4h timeframe (already aligned since using 4h data)
    supertrend_aligned = supertrend
    direction_aligned = direction
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite pivot level or trend reversal
        long_exit = close[i] < S3_aligned[i] or not uptrend
        short_exit = close[i] > R3_aligned[i] or not downtrend
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_breakout and downtrend and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 1d Williams %R for mean-reversion entries.
# Long when 12h Supertrend is bullish AND 1d Williams %R < -80 (oversold).
# Short when 12h Supertrend is bearish AND 1d Williams %R > -20 (overbought).
# Exit when Supertrend flips or Williams %R reverts to neutral (-50).
# Uses discrete position size 0.25. Combines trend-following (Supertrend) with mean-reversion (Williams %R)
# to capture trend-aligned pullbacks in both bull and bear markets. 6h timeframe targets 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data once before loop for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Supertrend (ATR=10, mult=3.0) ===
    # ATR calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    # Supertrend logic
    supertrend = np.zeros(len(close_12h))
    direction = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        # Recalculate if needed
        if direction[i] > 0 and supertrend[i] < supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
        if direction[i] < 0 and supertrend[i] > supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
    
    # === 1d Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align all indicators to primary timeframe (6h)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30  # Williams %R needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        supertrend_val = supertrend_aligned[i]
        direction_val = direction_aligned[i]
        williams_r_val = williams_r_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Supertrend turns bearish OR Williams %R > -50 (exits oversold)
            if (direction_val == -1) or (williams_r_val > -50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Supertrend turns bullish OR Williams %R < -50 (exits overbought)
            if (direction_val == 1) or (williams_r_val < -50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Supertrend bullish (direction=1) AND Williams %R < -80 (oversold)
            if (direction_val == 1) and (williams_r_val < -80):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Supertrend bearish (direction=-1) AND Williams %R > -20 (overbought)
            elif (direction_val == -1) and (williams_r_val > -20):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_12hSupertrend_1dWilliamsR_MeanReversion_V1"
timeframe = "6h"
leverage = 1.0
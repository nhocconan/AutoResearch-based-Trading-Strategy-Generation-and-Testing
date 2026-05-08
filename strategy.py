#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter + 1w Supertrend direction + 6h volume confirmation
# Long when CHOP(14) > 61.8 (ranging) AND price reverses from 6h Bollinger lower band (20,2) AND 1w Supertrend = bullish AND 6h volume > 1.5x 20-period average
# Short when CHOP(14) > 61.8 (ranging) AND price reverses from 6h Bollinger upper band (20,2) AND 1w Supertrend = bearish AND 6h volume > 1.5x 20-period average
# Exit when price crosses Bollinger middle band (20) or CHOP < 38.2 (trending regime)
# Designed for mean reversion in ranging markets with weekly trend filter to avoid counter-trend trades
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift

name = "6h_Chop_BBands_1wSupertrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Choppiness Index (14-period)
    atr_6h = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10((atr_6h * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # Default to middle range
    
    # 6h Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_middle = sma_20
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w Supertrend (ATR=10, mult=3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    upper_band = (high_1w + low_1w) / 2 + (3 * atr_1w)
    lower_band = (high_1w + low_1w) / 2 - (3 * atr_1w)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        # Update bands
        if close_1w[i-1] > upper_band[i-1]:
            upper_band[i] = max(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close_1w[i-1] < lower_band[i-1]:
            lower_band[i] = min(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Determine trend
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Supertrend direction: 1 = bullish, -1 = bearish
    supertrend_direction = direction
    
    # Align 1w Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(supertrend_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ranging market + price at BB lower + bullish weekly trend + volume spike
            long_cond = (chop[i] > 61.8) and (close[i] <= bb_lower[i]) and (supertrend_dir_aligned[i] == 1) and volume_filter[i]
            # Short conditions: ranging market + price at BB upper + bearish weekly trend + volume spike
            short_cond = (chop[i] > 61.8) and (close[i] >= bb_upper[i]) and (supertrend_dir_aligned[i] == -1) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses BB middle OR market starts trending
            if close[i] >= bb_middle[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses BB middle OR market starts trending
            if close[i] <= bb_middle[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
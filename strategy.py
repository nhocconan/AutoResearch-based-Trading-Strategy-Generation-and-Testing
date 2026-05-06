#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Bollinger Band breakout with volume confirmation and Choppiness Index regime filter
# Long when price breaks above upper Bollinger Band (20,2) with volume > 1.3x average and chop > 61.8 (range)
# Short when price breaks below lower Bollinger Band (20,2) with volume > 1.3x average and chop > 61.8 (range)
# Exits when price returns to middle Bollinger Band (20-period SMA)
# Designed to capture mean-reversion bounces in ranging markets while avoiding strong trends
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_12hBB20_2_Volume_ChopRange_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Bollinger Bands (20-period, 2 std)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_12h, middle_bb)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Calculate Choppiness Index (14-period) on 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop[np.isnan(chop) | (highest_high_14 - lowest_low_14 == 0)] = 50  # neutral when range=0
    
    # Range regime: chop > 61.8 (ranging market)
    range_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(range_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper BB with volume confirmation in ranging market
            if close[i] > upper_bb_aligned[i] and volume_filter[i] and range_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower BB with volume confirmation in ranging market
            elif close[i] < lower_bb_aligned[i] and volume_filter[i] and range_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB
            if close[i] >= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB
            if close[i] <= middle_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
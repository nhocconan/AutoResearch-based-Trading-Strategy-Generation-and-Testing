#!/usr/bin/env python3
"""
4h_1w_Camarilla_R1S1_Breakout_Volume_ATRFilter_v2
Hypothesis: Trade 4-hour chart using weekly Camarilla pivot R1/S1 breakouts with volume confirmation and ATR-based stop loss.
Weekly Camarilla levels provide strong weekly support/resistance. Breakouts with volume indicate institutional participation.
ATR stop loss manages risk during adverse moves. Works in bull/bear markets: breaks indicate momentum continuation.
Target: 20-50 total trades over 4 years (5-12/year) with position size 0.30.
"""

name = "4h_1w_Camarilla_R1S1_Breakout_Volume_ATRFilter_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Camarilla levels and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly average volume for spike detection (20-period)
    vol_1w = df_1w['volume'].values
    vol_avg_1w = np.full(len(vol_1w), np.nan)
    for i in range(len(vol_1w)):
        if i >= 19:  # 20-period average
            vol_avg_1w[i] = np.mean(vol_1w[i-19:i+1])
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Calculate ATR for stop loss (14-period on 4h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i >= 13:  # 14-period
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get previous completed weekly bar for Camarilla calculation
        if len(df_1w) < 2:
            continue
            
        # Calculate weekly Camarilla levels for each weekly bar
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Arrays to store weekly R1 and S1 levels
        weekly_r1 = np.full_like(weekly_close, np.nan)
        weekly_s1 = np.full_like(weekly_close, np.nan)
        
        # Calculate for each weekly bar (starting from index 1 to avoid look-ahead)
        for j in range(1, len(weekly_close)):
            range_val = weekly_high[j-1] - weekly_low[j-1]
            if range_val > 0:
                weekly_r1[j] = weekly_close[j-1] + (range_val * 1.1 / 12)
                weekly_s1[j] = weekly_close[j-1] - (range_val * 1.1 / 12)
        
        # Align the weekly R1/S1 to 4h timeframe
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_atr = atr[i]
        
        # Volume spike: current volume > 1.5x weekly average volume
        vol_spike = (not np.isnan(vol_avg_1w_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_1w_aligned[i])
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if (not np.isnan(weekly_r1_aligned[i]) and 
                current_close > weekly_r1_aligned[i] and vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = current_close
            # Short: price breaks below weekly S1 with volume spike
            elif (not np.isnan(weekly_s1_aligned[i]) and 
                  current_close < weekly_s1_aligned[i] and vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 or ATR stop loss
            if (not np.isnan(weekly_s1_aligned[i]) and 
                current_close < weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 or ATR stop loss
            if (not np.isnan(weekly_r1_aligned[i]) and 
                current_close > weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
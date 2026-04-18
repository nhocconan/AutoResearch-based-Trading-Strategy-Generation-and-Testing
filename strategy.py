#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volume confirmation. Enter long when price breaks above R1 with volume > 1.5x 20-period average and 1d EMA34 trending up. Enter short when price breaks below S1 with volume confirmation and 1d EMA34 trending down. Exit when price reverses to opposite Camarilla level (S1 for long, R1 for short) or trend changes. Target 20-40 trades/year via Camarilla breakout rarity + trend alignment + volume filter. Works in bull/bear by following 1d trend. Uses volume confirmation to avoid false breakouts.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.1*(high-low), 
    # R2 = close + high-low, R1 = close + 1.1*(high-low)/2, 
    # S1 = close - 1.1*(high-low)/2, S2 = close - (high-low), 
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)*1.1/2
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            rng = high[i-1] - low[i-1]
            camarilla_R1[i] = close[i-1] + 1.1 * rng / 2
            camarilla_S1[i] = close[i-1] - 1.1 * rng / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, vol_period, 1)  # need at least 1 for Camarilla (uses previous bar)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price > R1 + volume + 1d EMA trending up
            if (close[i] > camarilla_R1_aligned[i] and vol_confirm and 
                i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price < S1 + volume + 1d EMA trending down
            elif (close[i] < camarilla_S1_aligned[i] and vol_confirm and 
                  i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < S1 or 1d EMA turns down
            if close[i] < camarilla_S1_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > R1 or 1d EMA turns up
            if close[i] > camarilla_R1_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0
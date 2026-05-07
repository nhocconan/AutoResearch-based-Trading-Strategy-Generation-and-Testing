#!/usr/bin/env python3
"""
4h_HTF_Trend_With_Volume_Spike_Confirmation
Hypothesis: Trade in direction of higher timeframe trend (1d EMA50) with confirmation from 1w trend (price vs 1w EMA200) and volume spike. 
This avoids false breakouts by requiring alignment across multiple timeframes. Volume spike ensures institutional participation.
Designed for 20-30 trades/year to minimize fee drag. Works in bull/bear via trend filters.
"""

name = "4h_HTF_Trend_With_Volume_Spike_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for higher timeframe trend
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_1d_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Determine weekly trend
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_1w_up = close_1w_aligned[i] > ema_200_1w_aligned[i]
        trend_1w_down = close_1w_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: uptrend on both timeframes with volume spike
            if trend_1d_up and trend_1w_up and vol_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: downtrend on both timeframes with volume spike
            elif trend_1d_down and trend_1w_down and vol_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend breaks on either timeframe
            if not (trend_1d_up and trend_1w_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend breaks on either timeframe
            if not (trend_1d_down and trend_1w_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
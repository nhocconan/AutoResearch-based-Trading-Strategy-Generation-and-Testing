#!/usr/bin/env python3
"""
Hypothesis: Daily pivot points act as key support/resistance levels in crypto markets, with price often bouncing or breaking through them. 
This strategy uses the daily pivot point with volume confirmation and a 12h EMA filter to reduce false signals, targeting 20-30 trades per year.
Long entries occur when price closes above the daily pivot with volume > 1.5x average and price above the 12h EMA34.
Short entries occur when price closes below the daily pivot with volume > 1.5x average and price below the 12h EMA34.
Exits are triggered when price returns to the pivot level, avoiding whipsaw in ranging markets.
Designed for 4h timeframe to capture both breakout and mean-reversion moves in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot point (standard formula)
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily pivot and 12h EMA34 to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(ema_34_12h_4h[i]) or np.isnan(volume_ma_20.iloc[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price closes above pivot with volume spike and above 12h EMA34
            if price > pivot_4h[i] and vol > 1.5 * vol_ma and price > ema_34_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below pivot with volume spike and below 12h EMA34
            elif price < pivot_4h[i] and vol > 1.5 * vol_ma and price < ema_34_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to or below pivot
            if price <= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to or above pivot
            if price >= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Volume_EMA12h34"
timeframe = "4h"
leverage = 1.0
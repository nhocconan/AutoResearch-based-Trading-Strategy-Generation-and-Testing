#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_TrendFilter_v1
Concept: Donchian(20) breakout on 4h with volume confirmation and 1d EMA200 trend filter.
- Long when price breaks above 4h Donchian high (20) with volume >1.5x average and close > 1d EMA200
- Short when price breaks below 4h Donchian low (20) with volume >1.5x average and close < 1d EMA200
- Exit when price returns to 4h Donchian midpoint (mean of 20-period high/low)
- Uses EMA200 on daily timeframe for trend filter to avoid counter-trend trades
- Conservative sizing (0.25) to manage drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h: Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Get values
        ema200_val = ema200_1d_aligned[i]
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        donchian_mid_val = donchian_mid[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(donchian_mid_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume confirmation and above EMA200
            breakout_long = close_val > donchian_high_val
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_long and vol_confirm and close_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume confirmation and below EMA200
            elif close_val < donchian_low_val and vol_confirm and close_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below Donchian midpoint
            if close_val <= donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above Donchian midpoint
            if close_val >= donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above 1d Donchian upper band (20-period high) AND 1w EMA50 rising AND volume > 2.0x 20-period average.
Short when price breaks below 1d Donchian lower band (20-period low) AND 1w EMA50 falling AND volume > 2.0x 20-period average.
Exit when price crosses 1d Donchian middle band (20-period midpoint).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 7-25 trades/year per symbol.
1d timeframe avoids overtrading while capturing multi-day trends. 1w EMA50 provides strong trend filter with minimal lag.
Volume confirmation ensures only significant breakouts are taken. Donchian channels adapt to volatility.
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
    
    # Load 1d data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    donchian_Upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_Lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_Middle = (donchian_Upper + donchian_Lower) / 2.0
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and 1w indicators to 1d timeframe (prices are already 1d)
    donchian_Upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_Upper)
    donchian_Lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_Lower)
    donchian_Middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_Middle)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_Upper_aligned[i]) or np.isnan(donchian_Lower_aligned[i]) or 
            np.isnan(donchian_Middle_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian Upper AND 1w EMA50 rising AND volume spike
            if (price > donchian_Upper_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Lower AND 1w EMA50 falling AND volume spike
            elif (price < donchian_Lower_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian Middle band
            if position == 1 and price < donchian_Middle_aligned[i]:
                exit_signal = True
            elif position == -1 and price > donchian_Middle_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0
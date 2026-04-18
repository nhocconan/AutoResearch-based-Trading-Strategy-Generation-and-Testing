#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: Donchian channel breakouts on 4h timeframe with volume confirmation and 1d EMA200 trend filter capture strong directional moves while avoiding whipsaws in both bull and bear markets. 
Designed for ~25-35 trades/year to minimize fee drag and work across BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel on 4h (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for 1d EMA200 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        trend = ema_200_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume and uptrend
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and downtrend
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below lower Donchian or trend reverses
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above upper Donchian or trend reverses
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0
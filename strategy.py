#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_TrendFilter
Hypothesis: Buy when price breaks above 4h Donchian(20) high with volume spike and 1d EMA(50) uptrend.
Sell when price breaks below 4h Donchian(20) low with volume spike and 1d EMA(50) downtrend.
Uses tight entry conditions to target 20-50 trades/year, avoiding fee drag while capturing
strong trending moves. Works in bull markets via long breakouts and bear markets via
short breakdowns with trend filter preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema50 = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and uptrend
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and downtrend
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: break below Donchian low OR trend turns down
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: break above Donchian high OR trend turns up
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0
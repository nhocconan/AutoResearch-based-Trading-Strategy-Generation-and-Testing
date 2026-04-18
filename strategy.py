#!/usr/bin/env python3
"""
4h_Aggressive_Scalper_Breakout
Hypothesis: In range-bound markets, price often reverses at key intraday levels. This strategy uses 4-hour Donchian breakouts with volume confirmation and a trend filter (4h EMA50) to capture momentum bursts. It targets 20-40 trades per year per symbol, using small position sizes (0.25) to limit drawdown. The strategy is designed to work in both bull and bear markets by requiring volume confirmation and trend alignment, reducing false breakouts.
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
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h EMA50 trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need Donchian, volume MA, and EMA
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_spike = volume_spike[i]
        ema_val = ema_50[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and above EMA50
            if price > upper and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and below EMA50
            elif price < lower and vol_spike and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian or below EMA50
            if price < lower or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian or above EMA50
            if price > upper or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Aggressive_Scalper_Breakout"
timeframe = "4h"
leverage = 1.0
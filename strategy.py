#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Donchian(20) channel with volume spike and 1h EMA50 trend filter.
Captures breakouts in trending markets while avoiding false signals in chop. Uses volume to confirm breakout strength.
Target: 20-30 trades/year to minimize fee drag while capturing strong directional moves.
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
    
    # Donchian channel (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1h EMA50
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or
            np.isnan(ema_50_1h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        ema50 = ema_50_1h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and uptrend
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and downtrend
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below lower band OR trend turns down
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above upper band OR trend turns up
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
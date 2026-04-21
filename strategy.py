#!/usr/bin/env python3
"""
12h_Donchian_Breakout_With_Volume_Filter
Hypothesis: Enter long when price breaks above 12h Donchian(20) high with volume confirmation, 
short when price breaks below Donchian(20) low with volume confirmation. 
Exit on opposite Donchian break. This captures trend continuation with clear entry/exit rules. 
Volume filter ensures institutional participation. Designed for 12h timeframe to minimize 
trade frequency and maximize robustness in both bull and bear markets.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if price_close > upper_band and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation
            elif price_close < lower_band and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price breaks opposite Donchian band
            if position == 1 and price_close < lower_band:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0
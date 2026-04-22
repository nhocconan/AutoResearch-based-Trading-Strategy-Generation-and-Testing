#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 12h data once for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike filter (16-period average, approx 4 days)
    volume = prices['volume'].values
    vol_ma_16 = pd.Series(volume).rolling(window=16, min_periods=16).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any data is not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_16[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_16[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 16-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + volume spike + price > EMA34
            if price > upper and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + volume spike + price < EMA34
            elif price < lower and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through EMA or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below EMA or volume dries up
                if price < ema34 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above EMA or volume dries up
                if price > ema34 or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_20_EMA34_Volume_12h"
timeframe = "6h"
leverage = 1.0
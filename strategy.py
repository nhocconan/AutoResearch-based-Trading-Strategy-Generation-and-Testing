#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Trend Filter.
Long when price breaks above 20-period Donchian high + volume spike + price above 100-period SMA.
Short when price breaks below 20-period Donchian low + volume spike + price below 100-period SMA.
Exit on opposite Donchian breakout or when price crosses 100-period SMA.
Designed for 20-50 trades/year per symbol with strong trend-following edge.
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
    
    # 100-period SMA for trend filter (on close)
    sma_100 = pd.Series(close).rolling(window=100, min_periods=100).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0x average (to avoid false signals)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) and SMA (100)
    start_idx = max(20, 100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma_100[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current values
        upper = donchian_high[i]
        lower = donchian_low[i]
        sma_val = sma_100[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above upper band + volume filter + price above SMA
            if price_now > upper and vol_filter and price_now > sma_val:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume filter + price below SMA
            elif price_now < lower and vol_filter and price_now < sma_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below lower band OR price crosses below SMA
            if price_now < lower or price_now < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above upper band OR price crosses above SMA
            if price_now > upper or price_now > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
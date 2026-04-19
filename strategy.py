#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR-based trailing stop.
# Long when price breaks above 20-period high + volume > 1.8x 20-period average.
# Short when price breaks below 20-period low + volume > 1.8x 20-period average.
# Exit when price retraces to midpoint of Donchian channel or ATR stop is hit.
# Designed to capture trends with volume confirmation to avoid false breakouts.
# Target: 20-40 trades/year per symbol. Works in both bull (breakouts up) and bear (breakdowns down).
name = "4h_Donchian20_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ATR for volatility-based stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume confirmation
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume confirmation
            if price > donchian_high[i] and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short entry: price breaks below Donchian low + volume confirmation
            elif price < donchian_low[i] and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            
            # Exit conditions:
            # 1. Price retraces to midpoint of Donchian channel
            # 2. ATR-based trailing stop (3 * ATR from highest point)
            midpoint = donchian_mid[i]
            trailing_stop = highest_since_entry - 3.0 * atr[i]
            
            if price <= midpoint or price <= trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            
            # Exit conditions:
            # 1. Price retraces to midpoint of Donchian channel
            # 2. ATR-based trailing stop (3 * ATR from lowest point)
            midpoint = donchian_mid[i]
            trailing_stop = lowest_since_entry + 3.0 * atr[i]
            
            if price >= midpoint or price >= trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
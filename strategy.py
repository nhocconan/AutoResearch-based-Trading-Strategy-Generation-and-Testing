#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR stop
# - Uses 12h Donchian breakout for trend capture in both bull and bear markets
# - 1d volume filter ensures breakouts have institutional participation
# - ATR-based stop limits drawdown during adverse moves
# - Designed for low trade frequency (~15-25/year) to minimize fee drag
# - Works in bull markets by catching breakouts, in bear by shorting breakdowns

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR for stop loss (using 12h data)
    high_12h_arr = df_12h['high'].values
    low_12h_arr = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    tr1 = high_12h_arr - low_12h_arr
    tr2 = np.abs(high_12h_arr - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h_arr - np.roll(close_12h_arr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume surge
            if price > donch_high_12h[i] and vol > 1.5 * vol_ma_12h[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume surge
            elif price < donch_low_12h[i] and vol > 1.5 * vol_ma_12h[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR ATR stop hit (2*ATR)
            if price < donch_low_12h[i] or price < entry_price - 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR ATR stop hit (2*ATR)
            if price > donch_high_12h[i] or price > entry_price + 2.0 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0
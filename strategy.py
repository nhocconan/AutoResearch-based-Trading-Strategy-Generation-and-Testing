#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and ATR stop
# - Uses 1d Donchian (20-period) as trend filter: price must be above upper band for long, below lower for short
# - Entry: price breaks above/below 1d Donchian + volume > 1.3x 20-period average
# - Exit: price crosses back to opposite Donchian band or ATR-based stop (2x ATR from entry)
# - Volume confirmation reduces false breakouts
# - ATR stop manages risk during adverse moves
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(40, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or np.isnan(atr_1d_12h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume surge
            if price > donch_high_12h[i] and price > donch_high_12h[i-1] and vol > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume surge
            elif price < donch_low_12h[i] and price < donch_low_12h[i-1] and vol > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian low OR ATR stop hit (2*ATR)
            if price < donch_low_12h[i] or price < entry_price - 2.0 * atr_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high OR ATR stop hit (2*ATR)
            if price > donch_high_12h[i] or price > entry_price + 2.0 * atr_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0
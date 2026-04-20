#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR stop
# - Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average
# - Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average
# - Exit when price crosses back through opposite Donchian level or ATR stop hit
# - Uses 1d ATR for stop calculation (more stable) and 4h for entry/exit
# - Designed to work in both bull (breakouts) and bear (breakdowns) markets
# - Target: 20-30 trades per year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume surge
            if price > donch_high[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume surge
            elif price < donch_low[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian low OR ATR stop hit (2*ATR)
            if price < donch_low[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high OR ATR stop hit (2*ATR)
            if price > donch_high[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0
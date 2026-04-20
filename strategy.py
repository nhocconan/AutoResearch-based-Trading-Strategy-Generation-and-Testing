#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout above 1d VWAP with volume confirmation and ATR stop
# - Uses 1-day VWAP as trend filter: price must be above VWAP for long, below for short
# - Entry: price breaks above VWAP + volume > 2x 20-period average
# - Exit: price crosses back below VWAP or ATR-based stop hit (2.5x ATR)
# - VWAP is a strong institutional benchmark; breakouts often carry momentum
# - Volume surge confirms institutional participation
# - ATR stop manages risk during adverse moves
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    vwap_1d[vwap_denominator == 0] = np.nan
    
    # Align VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above VWAP + breaks above VWAP + volume surge
            if price > vwap_aligned[i] and price > vwap_aligned[i-1] and vol > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price below VWAP + breaks below VWAP + volume surge
            elif price < vwap_aligned[i] and price < vwap_aligned[i-1] and vol > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below VWAP OR ATR stop hit (2.5*ATR)
            if price < vwap_aligned[i] or price < entry_price - 2.5 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above VWAP OR ATR stop hit (2.5*ATR)
            if price > vwap_aligned[i] or price > entry_price + 2.5 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Breakout_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0
# 12h_Camarilla_R3S3_Volume_ATRStop - 12-hour timeframe with daily pivots
# Uses daily Camarilla pivot levels (R3/S3) for entry and R2/S2 for exit
# Adds volume confirmation (>1.5x 20-period average) and ATR-based stop (2x ATR)
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Works in both bull and bear markets via volatility-based breakouts

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Warmup period for daily calculations
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation (ONCE before loop)
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
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla pivot levels from prior day
    shift_high = np.roll(high_1d, 1)
    shift_low = np.roll(low_1d, 1)
    shift_close = np.roll(close_1d, 1)
    
    hl = shift_high - shift_low
    c = shift_close
    
    r3 = c + hl * 1.1 / 4
    s3 = c - hl * 1.1 / 4
    r2 = c + hl * 1.1 / 6
    s2 = c - hl * 1.1 / 6
    
    # Align pivot levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
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
        if np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + volume surge
            if price > r3_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S3 + volume surge
            elif price < s3_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below R2 OR ATR stop hit (2*ATR)
            if price < r2_12h[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above S2 OR ATR stop hit (2*ATR)
            if price > s2_12h[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0
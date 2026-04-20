#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h Supertrend filter and volume confirmation.
# Long when 4h Supertrend is bullish, price > 1h EMA20, and volume > 1.5x average.
# Short when 4h Supertrend is bearish, price < 1h EMA20, and volume > 1.5x average.
# Uses 4h Supertrend for trend direction (works in both bull/bear markets) and 1h for entry timing.
# Target: 15-30 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_4h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Align Supertrend direction to 1h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20 on 1h
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20, adjust=False).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(ema20[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        st_dir = supertrend_dir_aligned[i]
        ema20_val = ema20[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: 4h Supertrend bullish, price > EMA20, volume confirmation
            if st_dir == 1 and price > ema20_val and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: 4h Supertrend bearish, price < EMA20, volume confirmation
            elif st_dir == -1 and price < ema20_val and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h Supertrend turns bearish or price < EMA20
            if st_dir == -1 or price < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h Supertrend turns bullish or price > EMA20
            if st_dir == 1 or price > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_Supertrend_EMA20_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0
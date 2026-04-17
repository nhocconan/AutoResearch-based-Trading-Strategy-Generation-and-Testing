#!/usr/bin/env python3
"""
4h_HTF_Trend_LTF_Entry_v1
Trend follows 1d EMA34 direction with 4h Donchian breakout entries and ATR stop.
Long when: price > 1d EMA34 & breaks above 4h Donchian(20) high & volume > 1.5x MA(20)
Short when: price < 1d EMA34 & breaks below 4h Donchian(20) low & volume > 1.5x MA(20)
Exit: trailing stop at 3x ATR from extreme.
Designed to capture trends while avoiding whipsaw in ranges.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h Donchian(20) channels ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ATR(14) for volatility and stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume filter: volume > 1.5x 20-period MA ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position and extreme for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        if position == 0:
            # Look for entry signals
            long_entry = (close[i] > ema34_1d_aligned[i] and 
                         close[i] > donchian_high[i] and 
                         volume[i] > 1.5 * vol_ma[i])
            short_entry = (close[i] < ema34_1d_aligned[i] and 
                          close[i] < donchian_low[i] and 
                          volume[i] > 1.5 * vol_ma[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
                short_extreme = 0.0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
                long_extreme = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Update long extreme and check trailing stop
            long_extreme = max(long_extreme, high[i])
            trailing_stop = long_extreme - 3.0 * atr[i]
            
            if close[i] <= trailing_stop:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update short extreme and check trailing stop
            short_extreme = min(short_extreme, low[i])
            trailing_stop = short_extreme + 3.0 * atr[i]
            
            if close[i] >= trailing_stop:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Trend_LTF_Entry_v1"
timeframe = "4h"
leverage = 1.0
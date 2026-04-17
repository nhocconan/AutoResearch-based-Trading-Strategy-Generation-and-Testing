#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
Long when price breaks above upper Donchian channel AND volume > 1.5x 20-period average.
Short when price breaks below lower Donchian channel AND volume > 1.5x 20-period average.
Exit via ATR trailing stop (signal → 0 when price < highest high - 2*ATR for longs,
or price > lowest low + 2*ATR for shorts). Uses 1d EMA50 as trend filter (only long when price > EMA50,
only short when price < EMA50). Designed for low trade frequency (~20-50/year) on 4h timeframe
to minimize fee drag and work in both bull and bear markets via trend filter and volatility-based exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        ema50 = ema50_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price > upper Donchian AND volume > 1.5x avg AND price > 1d EMA50
            if price > upper and vol > 1.5 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price < lower Donchian AND volume > 1.5x avg AND price < 1d EMA50
            elif price < lower and vol > 1.5 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            
            # ATR trailing stop: exit if price < highest_high - 2*ATR
            if price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            
            # ATR trailing stop: exit if price > lowest_low + 2*ATR
            if price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRTrend_Filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_ATRStop_v1
Concept: 4h Donchian(20) breakout with 1d volume confirmation and ATR stop.
- Long when price breaks above 4h Donchian high(20) with above-average 1d volume
- Short when price breaks below 4h Donchian low(20) with above-average 1d volume
- Exit via ATR-based trailing stop (3x ATR from extreme)
- Works in bull/bear: Breakouts capture momentum, volume confirms institutional participation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 4h: Donchian channel (20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low with min_periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: ATR(14) for stop loss ===
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d: Volume average (20-period) for confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0  # For long trailing stop
    lowest_low = 0.0    # For short trailing stop
    
    start_idx = max(20, 14)  # Wait for Donchian and ATR
    
    for i in range(start_idx, n):
        # Get values
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma20_aligned[i]
        vol_1d_idx = i // 96  # Approximate 1d index from 4h (96 bars per day)
        
        # Skip if any value is NaN or invalid
        if (np.isnan(donch_high_val) or np.isnan(donch_low_val) or 
            np.isnan(atr_val) or np.isnan(vol_ma_val) or 
            vol_1d_idx >= len(vol_1d)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_1d_val = vol_1d[vol_1d_idx]
        vol_ma_1d_val = vol_ma20_1d[vol_1d_idx] if vol_1d_idx < len(vol_ma20_1d) else 0
        
        if position == 0:
            # Check for volume confirmation (1d volume above 20-day average)
            vol_confirm = vol_1d_val > vol_ma_1d_val and vol_ma_1d_val > 0
            
            # Long: Break above Donchian high with volume confirmation
            if close_val > donch_high_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_high = close_val
            # Short: Break below Donchian low with volume confirmation
            elif close_val < donch_low_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_low = close_val
        
        elif position == 1:
            # Update highest high for trailing stop
            highest_high = max(highest_high, close_val)
            # ATR trailing stop: exit if price drops 3*ATR from highest high
            if close_val <= highest_high - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, close_val)
            # ATR trailing stop: exit if price rises 3*ATR from lowest low
            if close_val >= lowest_low + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
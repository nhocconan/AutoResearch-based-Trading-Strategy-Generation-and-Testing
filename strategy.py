#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day ATR volatility filter and volume confirmation.
Long when price breaks above 20-period high with ATR > 1.5x average and volume > 1.5x average.
Short when price breaks below 20-period low with ATR > 1.5x average and volume > 1.5x average.
Exit when price returns to the 10-period moving average.
Designed for moderate trade frequency (~30-50/year) to capture volatility expansion moves.
Works in both bull and bear markets by requiring volatility expansion (ATR filter).
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
    
    # Load 1-day data for ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR calculation
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR average (20-period)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF ATR to lower timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_aligned[i]
        atr_ma_val = atr_ma_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ma_10_val = ma_10[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Break above Donchian high with volatility expansion and volume
            if (close_val > donchian_high_val and 
                atr_val > 1.5 * atr_ma_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volatility expansion and volume
            elif (close_val < donchian_low_val and 
                  atr_val > 1.5 * atr_ma_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to 10-period MA
                if close_val <= ma_10_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to 10-period MA
                if close_val >= ma_10_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Volume_Breakout"
timeframe = "4h"
leverage = 1.0
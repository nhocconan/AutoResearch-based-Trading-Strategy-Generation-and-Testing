#!/usr/bin/env python3
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
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range for ATR
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Calculate daily ATR(14)
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate daily ATR ratio (ATR(7)/ATR(14)) for volatility regime detection
    atr_7_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 7:
            atr_7_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_7_1d[i] = (atr_7_1d[i-1] * 6 + tr_1d[i]) / 7
    
    atr_ratio = np.divide(atr_7_1d, atr_1d, out=np.ones_like(atr_7_1d), where=atr_1d!=0)
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volatility filter: only trade when ATR ratio > 1.2 (expanding volatility)
    vol_expanding = atr_ratio_aligned > 1.2
    
    # Calculate daily Donchian channels (20-period)
    donchian_high_20 = np.full_like(close_1d, np.nan)
    donchian_low_20 = np.full_like(close_1d, np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_high_20[i] = np.max(high_1d[i-20:i])
        donchian_low_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_expanding[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volatility filter
        long_breakout = close[i] > donchian_high_aligned[i] and vol_expanding[i]
        short_breakout = close[i] < donchian_low_aligned[i] and vol_expanding[i]
        
        # Exit conditions: opposite breakout or volatility contraction
        exit_long = position == 1 and (close[i] < donchian_low_aligned[i] or not vol_expanding[i])
        exit_short = position == -1 and (close[i] > donchian_high_aligned[i] or not vol_expanding[i])
        
        # Execute signals
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_breakout_vol_expansion_v1"
timeframe = "4h"
leverage = 1.0
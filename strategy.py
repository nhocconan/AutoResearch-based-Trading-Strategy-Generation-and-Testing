#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian Channel (20-period)
    donch_high_1d = np.full(len(df_1d), np.nan)
    donch_low_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high_1d[i] = np.max(high_1d[i-19:i+1])
        donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 1d ATR moving average (20-period)
    atr_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(33, len(df_1d)):  # 14 + 19 for 20-period MA
        atr_ma_20_1d[i] = np.mean(atr_1d[i-19:i+1])
    
    # Align 1d indicators to 4h timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # Calculate 4h ATR (14-period) for position sizing and stop
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.mean(tr_h[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility regimes
        vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma_20_1d_aligned[i]
        
        # Breakout conditions: price breaks 1d Donchian levels
        breakout_long = close[i] > donch_high_1d_aligned[i]
        breakout_short = close[i] < donch_low_1d_aligned[i]
        
        # Entry conditions: breakout + volatility filter
        long_entry = breakout_long and vol_filter
        short_entry = breakout_short and vol_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = breakout_short or (atr_1d_aligned[i] < 0.3 * atr_ma_20_1d_aligned[i])
        short_exit = breakout_long or (atr_1d_aligned[i] < 0.3 * atr_ma_20_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0
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
    
    # Get 1w and 1d data for context
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high_1w = np.full(len(df_1w), np.nan)
    donchian_low_1w = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donchian_high_1w[i] = np.max(high_1w[i-19:i+1])
        donchian_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate 1d ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align weekly and daily indicators to 12h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_12h = np.full(n, np.nan)
    for i in range(14, n):
        atr_12h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 12h volume moving average
    vol_s_h = pd.Series(volume)
    vol_ma_20_h = vol_s_h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_20_h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR(1d) to avoid low volatility
        vol_filter = atr_12h[i] > 0.5 * atr_1d_aligned[i]
        
        # Volume filter: volume > 1.5 * 20-period MA
        vol_spike = volume[i] > 1.5 * vol_ma_20_h[i]
        
        # Breakout conditions: price breaks weekly Donchian channels
        long_breakout = close[i] > donchian_high_1w_aligned[i]
        short_breakout = close[i] < donchian_low_1w_aligned[i]
        
        # Entry conditions: breakout with volatility and volume filters
        long_entry = long_breakout and vol_filter and vol_spike
        short_entry = short_breakout and vol_filter and vol_spike
        
        # Exit conditions: price returns to middle of Donchian channel
        mid_channel = (donchian_high_1w_aligned[i] + donchian_low_1w_aligned[i]) / 2
        long_exit = close[i] < mid_channel
        short_exit = close[i] > mid_channel
        
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

name = "12h_1w_donchian_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0
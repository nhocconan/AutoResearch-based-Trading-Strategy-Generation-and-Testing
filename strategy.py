#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w/1d pivot direction filter and volume confirmation
    # Donchian breakouts capture momentum bursts; 1w pivot defines major trend bias; 1d pivot refines entry timing
    # Volume >1.5x 20-period average confirms institutional participation
    # Target: 12-30 trades/year (50-120 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for pivot entry levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w Camarilla H3/L3 for trend bias (based on previous 1w bar)
    camarilla_h3_1w = np.full(len(high_1w), np.nan)
    camarilla_l3_1w = np.full(len(low_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        rang = ph - pl
        
        camarilla_h3_1w[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3_1w[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Calculate 1d Camarilla H3/L3 for entry levels (based on previous 1d bar)
    camarilla_h3_1d = np.full(len(high_1d), np.nan)
    camarilla_l3_1d = np.full(len(low_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        camarilla_h3_1d[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3_1d[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Get 6h Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (1.5 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    donchian_high_aligned = donchian_high  # already LTF
    donchian_low_aligned = donchian_low    # already LTF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w pivot bias: price above H3 = bullish bias, below L3 = bearish bias
        bullish_bias = close[i] > h3_1w_aligned[i]
        bearish_bias = close[i] < l3_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic: Breakout in direction of 1w bias + 1d pivot confirmation + volume
        long_entry = long_breakout and bullish_bias and (close[i] > h3_1d_aligned[i]) and volume_spike_6h[i]
        short_entry = short_breakout and bearish_bias and (close[i] < l3_1d_aligned[i]) and volume_spike_6h[i]
        
        # Exit logic: price retests opposite Donchian channel or 1w bias reversal
        long_exit = (close[i] <= donchian_low_aligned[i]) or (not bullish_bias)
        short_exit = (close[i] >= donchian_high_aligned[i]) or (not bearish_bias)
        
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

name = "6h_1w_1d_donchian_breakout_pamir_v1"
timeframe = "6h"
leverage = 1.0
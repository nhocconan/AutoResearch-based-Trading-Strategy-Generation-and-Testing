#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Uses 1d Camarilla H3/L3 levels for breakout entries
    # 1d volume spike (>2.0x 20-period average) confirms breakout strength
    # 12h chopiness index > 61.8 ensures ranging market for mean reversion at extremes
    # Designed for low trade frequency (target: 15-30/year) to minimize fee drag
    # Works in bull/bear markets by fading extremes in ranging conditions
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3) from previous day
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        high_low = high_1d[i-1] - low_1d[i-1]
        camarilla_h3[i] = close_1d[i-1] + 1.1 * high_low / 2
        camarilla_l3[i] = close_1d[i-1] - 1.1 * high_low / 2
    
    # Get 1d volume for confirmation
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Volume spike: volume > 2.0 * 20-period average (1d)
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Calculate 12h chopiness index for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    # ATR(14)
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.mean(tr[i-13:i+1])
    
    # Chopiness Index: 100 * log10(sum(atr/14) / (max(high)-min(low))) / log10(14)
    chop = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_sum = np.sum(atr_12h[i-13:i+1])
        max_high = np.max(high_12h[i-13:i+1])
        min_low = np.min(low_12h[i-13:i+1])
        if max_high > min_low:
            chop[i] = 100 * np.log10(atr_sum / 14 / (max_high - min_low)) / np.log10(14)
    
    # Chop > 61.8 = ranging market (good for mean reversion)
    chop_ranging = chop > 61.8
    
    # Align all indicators to LTF
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_12h, chop_ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_ranging_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: price breaks Camarilla H3/L3 + volume spike + chop ranging
        long_entry = False
        short_entry = False
        
        # Long: price breaks above H3 (short-term mean reversion fade)
        if close[i] > camarilla_h3_aligned[i]:
            long_entry = volume_spike_aligned[i] and chop_ranging_aligned[i]
        # Short: price breaks below L3 (short-term mean reversion fade)
        elif close[i] < camarilla_l3_aligned[i]:
            short_entry = volume_spike_aligned[i] and chop_ranging_aligned[i]
        
        # Exit logic: price returns to midpoint or opposite extreme
        camarilla_mid = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        
        long_exit = close[i] < camarilla_mid
        short_exit = close[i] > camarilla_mid
        
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

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0
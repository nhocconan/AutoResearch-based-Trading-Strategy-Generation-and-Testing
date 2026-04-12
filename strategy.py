#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla breakout with 1d ATR filter and volume confirmation
    # Uses 1d Camarilla levels for structure, 1d ATR for volatility filter, volume spike for confirmation
    # Designed for low trade frequency (target: 20-40/year) to minimize fee drag
    # ATR filter avoids whipsaws in low volatility, works in bull/bear via breakout structure
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla pivot levels (L3/H3 for entries)
    camarilla_h3_1d = np.full(len(df_1d), np.nan)
    camarilla_l3_1d = np.full(len(df_1d), np.nan)
    pivot_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        pivot_1d[i] = pivot_val
        camarilla_h3_1d[i] = pivot_val + range_val * 1.1 / 4.0
        camarilla_l3_1d[i] = pivot_val - range_val * 1.1 / 4.0
    
    # Calculate 1d ATR(14) for volatility filter
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr1 = high_1d[i] - low_1d[i]
        tr2 = abs(high_1d[i] - close_1d[i-1])
        tr3 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr1, tr2, tr3)
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Align Camarilla levels and ATR to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average (4h) for stricter filter
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is elevated (avoid choppy low vol periods)
        atr_ratio = atr_1d_aligned[i] / (np.mean(atr_1d_aligned[max(0, i-50):i+1]) + 1e-10)
        high_volatility = atr_ratio > 1.2
        
        # Entry logic: Camarilla breakout with volume and volatility filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above camarilla H3 with volume spike and high volatility
        if high_volatility:
            long_entry = (close[i] > camarilla_h3_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below camarilla L3 with volume spike and high volatility
        elif high_volatility:
            short_entry = (close[i] < camarilla_l3_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite camarilla level
        long_exit = close[i] < camarilla_l3_aligned[i]
        short_exit = close[i] > camarilla_h3_aligned[i]
        
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

name = "4h_1d_camarilla_breakout_atr_volume_v1"
timeframe = "4h"
leverage = 1.0
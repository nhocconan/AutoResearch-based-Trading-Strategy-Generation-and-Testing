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
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels on 4h
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === 1d ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 4h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure indicators have enough data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        atr = atr_4h_aligned[i]
        
        # === STOPLOSS: 2 ATR from entry ===
        if position == 1:  # Long position
            if price <= entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price >= entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC: Mean reversion to middle of channel ===
        if position == 1:  # Long position
            # Exit when price crosses below midline
            midline = (dh_4h_aligned[i] + dl_4h_aligned[i]) / 2
            if price < midline:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above midline
            midline = (dh_4h_aligned[i] + dl_4h_aligned[i]) / 2
            if price > midline:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation
            if price > dh_4h_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation
            elif price < dl_4h_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Camarilla pivot levels (R4/S4) and volume confirmation.
# Uses wider bands (R4/S4) for fewer, higher-quality trades. Long: price crosses above R4 + volume > 1.5x avg volume (20).
# Short: price crosses below S4 + volume > 1.5x avg volume. Target: 50-150 total trades over 4 years (12-37/year).
# Wider bands reduce frequency while maintaining edge in ranging and trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R4/S4 - wider bands for fewer trades)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        # Camarilla R4 and S4 levels (wider bands)
        camarilla_r4[i] = pc + (rang * 1.1 / 2)  # R4 = C + 1.1*range/2
        camarilla_s4[i] = pc - (rang * 1.1 / 2)  # S4 = C - 1.1*range/2
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Camarilla levels to 12h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price crosses above R4 + volume confirmation
            if (price > r4 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below S4 + volume confirmation
            elif (price < s4 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S4 (opposite level)
            if price < s4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above R4 (opposite level)
            if price > r4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_R4S4_Volume"
timeframe = "12h"
leverage = 1.0
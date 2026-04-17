#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_V1
Long/short at Camarilla R1/S1 with volume confirmation and 12h trend filter.
Exit at R4/S4 or opposite S1/R1.
Designed to capture reversals at key intraday levels with institutional volume.
Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Camarilla levels from previous day ===
    # Use daily high/low/close from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    # Align to 4h timeframe
    ph_aligned = align_htf_to_ltf(prices, df_1d, ph)
    pl_aligned = align_htf_to_ltf(prices, df_1d, pl)
    pc_aligned = align_htf_to_ltf(prices, df_1d, pc)
    
    # Calculate Camarilla levels
    R1 = pc_aligned + 1.1 * (ph_aligned - pl_aligned) / 12
    S1 = pc_aligned - 1.1 * (ph_aligned - pl_aligned) / 12
    R4 = pc_aligned + 1.1 * (ph_aligned - pl_aligned) / 2
    S4 = pc_aligned - 1.1 * (ph_aligned - pl_aligned) / 2
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h trend filter: EMA50 ===
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(R4[i]) or np.isnan(S4[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: require volume > 1.5x average
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price crosses above R1 with volume, above 12h EMA50
            if (close[i] > R1[i] and close[i-1] <= R1[i-1] and vol_ok and
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price crosses below S1 with volume, below 12h EMA50
            elif (close[i] < S1[i] and close[i-1] >= S1[i-1] and vol_ok and
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches R4 or crosses below S1
            if (close[i] >= R4[i] or close[i] < S1[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S4 or crosses above R1
            if (close[i] <= S4[i] or close[i] > R1[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_V1"
timeframe = "4h"
leverage = 1.0
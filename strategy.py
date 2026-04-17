#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (more precise than standard)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    hl_range = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = close_1d + (hl_range * 1.1 / 2)
    r3 = close_1d + (hl_range * 1.1 / 4)
    r2 = close_1d + (hl_range * 1.1 / 6)
    r1 = close_1d + (hl_range * 1.1 / 12)
    s1 = close_1d - (hl_range * 1.1 / 12)
    s2 = close_1d - (hl_range * 1.1 / 6)
    s3 = close_1d - (hl_range * 1.1 / 4)
    s4 = close_1d - (hl_range * 1.1 / 2)
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_prev)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Volume confirmation: current volume > 1.3 * 20-period average (to reduce noise)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_4h[i]) or 
            np.isnan(s2_4h[i]) or
            np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average (moderate filter)
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and volatility
            if close[i] > r2_4h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and volatility
            elif close[i] < s2_4h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops significantly
            if close[i] < r1_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops significantly
            if close[i] > s1_4h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_Vol_v4"
timeframe = "4h"
leverage = 1.0
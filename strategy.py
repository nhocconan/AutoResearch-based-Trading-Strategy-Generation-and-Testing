#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_v3
Hypothesis: Daily Camarilla pivot levels (R4/S4) with volume confirmation and 
4h ATR volatility filter. Long when price breaks above R4 with volume spike and 
ATR < median ATR (low volatility). Short when price breaks below S4 with volume 
spike and ATR < median ATR. Uses 1d Camarilla for structure, volume confirmation 
for conviction, and ATR filter to avoid whipsaws. Targets 50-150 total trades 
over 4 years to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2]
    prev_low = df_1d['low'].iloc[-2]
    prev_close = df_1d['close'].iloc[-2]
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    if range_ <= 0:
        return np.zeros(n)
    
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Create arrays for each day's levels
    camarilla_r4_array = np.full(len(df_1d), camarilla_r4)
    camarilla_s4_array = np.full(len(df_1d), camarilla_s4)
    
    # Align to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_array)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_array)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR filter: only trade when ATR < median ATR (low volatility)
    # Calculate ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Median ATR for filtering
    median_atr = np.nanmedian(atr[14:]) if np.sum(~np.isnan(atr[14:])) > 0 else np.inf
    atr_filter = atr < median_atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and ATR filter
        long_breakout = high[i] > camarilla_r4_aligned[i] and volume_spike[i] and atr_filter[i]
        short_breakout = low[i] < camarilla_s4_aligned[i] and volume_spike[i] and atr_filter[i]
        
        # Exit conditions: return to Camarilla midpoint (mean reversion)
        camarilla_midpoint = (camarilla_r4 + camarilla_s4) / 2
        camarilla_midpoint_array = np.full(len(df_1d), camarilla_midpoint)
        camarilla_midpoint_aligned = align_htf_to_ltf(prices, df_1d, camarilla_midpoint_array)
        
        long_exit = close[i] < camarilla_midpoint_aligned[i]
        short_exit = close[i] > camarilla_midpoint_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
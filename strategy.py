#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike
# Hypothesis: Camarilla pivot breakouts at R1/S1 with daily trend filter and volume spike capture institutional moves in both bull and bear markets.
# Timeframe: 4h, uses 1d trend filter for multi-timeframe alignment.
# Low trade frequency (~20-30/year) via strict R1/S1 breakout + volume + trend confluence.
# Long: Close > R1 with volume > 2x average and daily uptrend (close > EMA34).
# Short: Close < S1 with volume > 2x average and daily downtrend (close < EMA34).
# Exit: Opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend failure.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
leverage = 1.0

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
    
    # Camarilla levels from previous day (using daily OHLC)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # We'll calculate these from daily data and align to 4h
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily timeframe
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Camarilla R1 and S1
    camarilla_r1 = d_close + 1.1 * (d_high - d_low) / 12
    camarilla_s1 = d_close - 1.1 * (d_high - d_low) / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day after calculation)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike and daily uptrend
            if close[i] > camarilla_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume spike and daily downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
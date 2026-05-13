#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as key support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and alignment to daily trend (via EMA34) 
capture institutional moves. Works in bull markets by catching breakouts and in bear markets by 
fading false breaks or catching reversals with volume divergence.
"""

name = "12h_1D_Camarilla_R1_S1_Breakout"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 34 for EMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_r1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        camarilla_s1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = np.zeros_like(close_1d)
    ema34_1d[:] = np.nan
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])  # Simple average for first value
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align daily indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and price > daily EMA34
            if (close[i] > camarilla_r1_aligned[i] and vol_spike and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and price < daily EMA34
            elif (close[i] < camarilla_s1_aligned[i] and vol_spike and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (mean reversion) or loss of volume
            if (close[i] < camarilla_s1_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (mean reversion) or loss of volume
            if (close[i] > camarilla_r1_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
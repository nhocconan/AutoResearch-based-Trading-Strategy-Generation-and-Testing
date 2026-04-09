#!/usr/bin/env python3
# 12h_trix_volume_sr_1w
# Hypothesis: TRIX (12) on weekly timeframe for momentum direction, combined with 12h price crossing 
# dynamic support/resistance (EMA20) and volume confirmation. Designed to capture medium-term 
# momentum shifts in both bull and bear markets with strict entry conditions to limit trades.
# Target: 15-35 trades/year (60-140 total over 4 years) with low turnover.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_sr_1w"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. Weekly TRIX (12-period) for momentum direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    # Calculate TRIX: EMA(EMA(EMA(close, 12), 12), 12) then % change
    close_1w = df_1w['close'].values
    ema1 = np.zeros(len(close_1w))
    ema2 = np.zeros(len(close_1w))
    ema3 = np.zeros(len(close_1w))
    
    # First EMA
    ema1[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema1[i] = (close_1w[i] * 2 / (12 + 1)) + (ema1[i-1] * (1 - 2 / (12 + 1)))
    
    # Second EMA
    ema2[0] = ema1[0]
    for i in range(1, len(close_1w)):
        ema2[i] = (ema1[i] * 2 / (12 + 1)) + (ema2[i-1] * (1 - 2 / (12 + 1)))
    
    # Third EMA
    ema3[0] = ema2[0]
    for i in range(1, len(close_1w)):
        ema3[i] = (ema2[i] * 2 / (12 + 1)) + (ema3[i-1] * (1 - 2 / (12 + 1)))
    
    # TRIX = % change of triple EMA
    trix = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # 2. Dynamic support/resistance: 20-period EMA on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = np.zeros(len(close_1d))
    
    # Calculate EMA20
    ema20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema20_1d[i] = (close_1d[i] * 2 / (20 + 1)) + (ema20_1d[i-1] * (1 - 2 / (20 + 1)))
    
    # Align EMA20 to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 3. Volume confirmation: 20-period average volume on 12h
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative OR price closes below EMA20
            if trix_aligned[i] < 0 or close[i] < ema20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns positive OR price closes above EMA20
            if trix_aligned[i] > 0 or close[i] > ema20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TRIX positive AND price above EMA20 with volume confirmation
            if (trix_aligned[i] > 0 and 
                close[i] > ema20_1d_aligned[i] and 
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: TRIX negative AND price below EMA20 with volume confirmation
            elif (trix_aligned[i] < 0 and 
                  close[i] < ema20_1d_aligned[i] and 
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_1D_VOLUME
Hypothesis: Use 4h EMA50 and 1d EMA200 for trend direction, 1h for entry timing with Camarilla R1/S1 breakout and volume confirmation (1.5x). Reduces trades by requiring multi-timeframe alignment. Target: 15-30 trades/year.
"""
name = "1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_1D_VOLUME"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 1d data for Camarilla calculation and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA200 on 1d for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Camarilla R1 and S1 levels from previous 1d bar
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        camarilla_r1[i] = pc + range_val * 1.1 / 6
        camarilla_s1[i] = pc - range_val * 1.1 / 6
    
    # Volume spike: current 1h volume > 1.5x 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # Align all HTF data to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike in uptrend (4h and 1d aligned)
            if (high[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_4h_aligned[i] and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume spike in downtrend (4h and 1d aligned)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend reversal (4h or 1d)
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema50_4h_aligned[i] or 
                close[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend reversal (4h or 1d)
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema50_4h_aligned[i] or 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla pivot points once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n1d = len(close_1d)
    camarilla_H4 = np.full(n1d, np.nan)
    camarilla_L4 = np.full(n1d, np.nan)
    camarilla_H3 = np.full(n1d, np.nan)
    camarilla_L3 = np.full(n1d, np.nan)
    camarilla_H2 = np.full(n1d, np.nan)
    camarilla_L2 = np.full(n1d, np.nan)
    camarilla_H1 = np.full(n1d, np.nan)
    camarilla_L1 = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        PH = high_1d[i-1]
        PL = low_1d[i-1]
        PC = close_1d[i-1]
        
        RANGE = PH - PL
        camarilla_H4[i] = PC + 1.5 * RANGE
        camarilla_L4[i] = PC - 1.5 * RANGE
        camarilla_H3[i] = PC + 1.25 * RANGE
        camarilla_L3[i] = PC - 1.25 * RANGE
        camarilla_H2[i] = PC + 1.08333 * RANGE
        camarilla_L2[i] = PC - 1.08333 * RANGE
        camarilla_H1[i] = PC + 1.08333 * (RANGE / 2)
        camarilla_L1[i] = PC - 1.08333 * (RANGE / 2)
    
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_H1_aligned[i]) or 
            np.isnan(camarilla_L1_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H1 with 4h uptrend + volume spike
            long_cond = (close[i] > camarilla_H1_aligned[i] and 
                        ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below L1 with 4h downtrend + volume spike
            short_cond = (close[i] < camarilla_L1_aligned[i] and 
                         ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below L1
            if close[i] < camarilla_L1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H1
            if close[i] > camarilla_H1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
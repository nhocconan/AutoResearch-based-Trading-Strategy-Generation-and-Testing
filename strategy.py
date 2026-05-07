#!/usr/bin/env python3
# 1h_1D_CAMARILLA_4H_TREND_FILTER
# Hypothesis: Use 1D Camarilla pivot breakout with 4H EMA trend filter for direction, 
# enter on 1H bar close with volume confirmation. Designed for low trade frequency 
# to minimize fee drag and work in both bull/bear markets by capturing momentum 
# from institutional pivot levels.
timeframe = "1h"
name = "1h_1D_Camarilla_4H_Trend_Filter"
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
    
    # Get daily data for Camarilla levels (calculate once)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla levels
    camarilla_h5 = pc + 1.1 * (ph - pl) / 2
    camarilla_h4 = pc + 1.1 * (ph - pl) / 4
    camarilla_h3 = pc + 1.1 * (ph - pl) / 6
    camarilla_l3 = pc - 1.1 * (ph - pl) / 6
    camarilla_l4 = pc - 1.1 * (ph - pl) / 4
    camarilla_l5 = pc - 1.1 * (ph - pl) / 2
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4H EMA50 for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF data to 1H timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above Camarilla H4 + 4H EMA50 uptrend + volume
            if (close[i] > camarilla_h4_aligned[i] and 
                ema_4h_aligned[i] > ema_4h_aligned[i-1] and  # 4H EMA rising
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price closes below Camarilla L4 + 4H EMA50 downtrend + volume
            elif (close[i] < camarilla_l4_aligned[i] and 
                  ema_4h_aligned[i] < ema_4h_aligned[i-1] and  # 4H EMA falling
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price closes below Camarilla H3 or 4H EMA turns down
            if (close[i] < camarilla_h3_aligned[i] or 
                ema_4h_aligned[i] < ema_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price closes above Camarilla L3 or 4H EMA turns up
            if (close[i] > camarilla_l3_aligned[i] or 
                ema_4h_aligned[i] > ema_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
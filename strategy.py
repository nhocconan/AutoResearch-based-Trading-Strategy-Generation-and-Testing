#!/usr/bin/env python3
# 12h_1d_camarilla_volume_v1
# Strategy: 12h Camarilla pivot levels with 1d volume confirmation and EMA trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price reversals at Camarilla pivot levels (L3, H3) with volume confirmation and trend alignment.
# Long when price crosses above L3 with volume > 1.5x average and price > 50 EMA.
# Short when price crosses below H3 with volume > 1.5x average and price < 50 EMA.
# Designed for low frequency (12-37 trades/year) to minimize fee drag in ranging and trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d close for Camarilla calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    # H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    daily_range = high_1d - low_1d
    camarilla_H4 = close_1d + 1.1 * daily_range / 2
    camarilla_L4 = close_1d - 1.1 * daily_range / 2
    camarilla_H3 = close_1d + 1.1 * daily_range / 4
    camarilla_L3 = close_1d - 1.1 * daily_range / 4
    camarilla_H2 = close_1d + 1.1 * daily_range / 6
    camarilla_L2 = close_1d - 1.1 * daily_range / 6
    camarilla_H1 = close_1d + 1.1 * daily_range / 12
    camarilla_L1 = close_1d - 1.1 * daily_range / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_L3_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * volume_ma.iloc[i]
        
        # Entry conditions
        if volume_confirm and close[i] > camarilla_L3_aligned[i] and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif volume_confirm and close[i] < camarilla_H3_aligned[i] and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite signal or trend reversal
        elif position == 1 and (close[i] < camarilla_L3_aligned[i] or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_H3_aligned[i] or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
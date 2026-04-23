#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 Breakout with 12h EMA50 Trend Filter and Volume Spike
- Uses Camarilla pivot levels (R1/S1) from 4h timeframe for breakout signals
- 12h EMA50 defines higher timeframe trend filter: only trade breakouts in direction of higher timeframe trend
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Exit when price returns to Camarilla pivot point (PP) or trend reverses
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in both bull and bear markets by trading breakouts with higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 4h
    # PP = (High + Low + Close) / 3
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    typical_price = (high + low + close) / 3.0
    price_range = high - low
    pp = typical_price
    r1 = close + price_range * 1.1 / 12.0
    s1 = close - price_range * 1.1 / 12.0
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 AND price above 12h EMA50 AND volume spike
            if (close[i] > r1[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 AND price below 12h EMA50 AND volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close returns to PP OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when close <= PP OR price closes below 12h EMA50
                if (close[i] <= pp[i] or close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when close >= PP OR price closes above 12h EMA50
                if (close[i] >= pp[i] or close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
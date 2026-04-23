#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA200 Trend Filter and Volume Confirmation
- Uses Camarilla pivot levels (R1/S1) from 4h timeframe for structure-based entries
- 4h EMA200 defines higher timeframe trend filter: only trade in direction of 4h trend
- Volume confirmation (> 1.5x 20-period average) filters weak signals
- Exit when price retouches Camarilla pivot point (PP) or trend reverses
- Session filter: 08-20 UTC to avoid low liquidity periods
- Target: 15-37 trades/year (60-150 over 4 years) by using 4h for direction and 1h for timing
- Works in both bull and bear markets by combining breakout logic with trend filter
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
    
    # Calculate 4h Camarilla pivot levels (R1/S1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations: based on previous 4h bar's range
    PP = (high_4h + low_4h + close_4h) / 3
    R = high_4h - low_4h
    R1 = PP + R * 1.1 / 12
    S1 = PP - R * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    PP_aligned = align_htf_to_ltf(prices, df_4h, PP)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Calculate 4h EMA200 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_200_4h = close_4h_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 200)  # for volume MA and EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 4h EMA200 AND volume spike AND in session
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_200_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND below 4h EMA200 AND volume spike AND in session
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_200_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price retouches PP OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches PP OR closes below 4h EMA200
                if (close[i] <= PP_aligned[i] or close[i] < ema_200_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches PP OR closes above 4h EMA200
                if (close[i] >= PP_aligned[i] or close[i] > ema_200_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA200_Trend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0
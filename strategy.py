#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation
# Hypothesis: Use 1d OHLC to calculate Camarilla R1/S1 levels for institutional support/resistance.
# Breakout above R1 or below S1 with volume confirmation signals institutional interest.
# Use 4h trend filter (price above/below 4h EMA50) to align with higher timeframe momentum.
# Entry timing on 1h with session filter (08-20 UTC) to avoid low-volume Asian session.
# Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.

name = "1h_4h_1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 1h timeframe (waits for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_4h_50_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume confirmation and 4h uptrend
            if (close[i] > camarilla_r1_aligned[i] and volume_confirm[i] and 
                close[i] > ema_4h_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume confirmation and 4h downtrend
            elif (close[i] < camarilla_s1_aligned[i] and volume_confirm[i] and 
                  close[i] < ema_4h_50_aligned[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or 4h trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or 4h trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
# Hypothesis: 12h Camarilla Pivot (R1/S1) breakout with 1d trend filter and volume confirmation.
# Uses Camarilla pivot levels from the previous 1d bar to identify breakouts.
# Breakouts above R1 or below S1 trigger entries in the direction of the 1d EMA(34) trend.
# Volume confirmation (1.5x 20-period average) filters false breakouts.
# Designed for 12h timeframe with ~12-37 trades per year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: P = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate pivot and levels
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (they are constant for the bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: breakout from Camarilla levels in trend direction with volume
        long_breakout = close[i] > camarilla_r1_aligned[i]
        short_breakout = close[i] < camarilla_s1_aligned[i]
        
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: price returns to pivot or trend reversal
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
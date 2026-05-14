#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_camarilla_pivot_breakout_volume
# Uses daily Camarilla pivot levels (from 1d data) on 6h chart.
# Long when price breaks above R4 with volume confirmation (volume > 2.0x 20-period avg).
# Short when price breaks below S4 with volume confirmation.
# Exits when price returns to the 1d close (mean reversion to daily close).
# Volume confirmation reduces false breakouts.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and via mean reversion to daily close in ranging markets.

name = "6h_1d_camarilla_pivot_breakout_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    
    # First value will be invalid due to roll, set to NaN
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_R4 = close_1d_prev + 1.5 * (high_1d_prev - low_1d_prev)
    camarilla_S4 = close_1d_prev - 1.5 * (high_1d_prev - low_1d_prev)
    # Exit level: previous day's close
    camarilla_exit = close_1d_prev
    
    # Align daily Camarilla levels to 6h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_exit_aligned = align_htf_to_ltf(prices, df_1d, camarilla_exit)
    
    # Volume confirmation: volume > 2.0 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or 
            np.isnan(camarilla_exit_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above R4
        if close[i] > camarilla_R4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below S4
        elif close[i] < camarilla_S4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to previous day's close (mean reversion)
        elif position == 1 and close[i] <= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
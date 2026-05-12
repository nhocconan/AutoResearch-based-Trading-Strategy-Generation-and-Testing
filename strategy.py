#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade Camarilla pivot level breakouts (R1/S1) on 4h with 1d EMA trend filter and volume confirmation.
# Long when price breaks above R1, above 1d EMA34, and volume > 1.5x average. Short when price breaks below S1, below 1d EMA34, and volume > 1.5x average.
# Exit on opposite breakout or trend failure. Designed for 20-50 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for volume filter (50-period)
    avg_vol = pd.Series(volume).rolling(window=50, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure volume average is stable
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for previous period
        # Use previous bar's high/low/close to avoid look-ahead
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla levels
        range_val = ph - pl
        if range_val <= 0:
            r1 = s1 = pc
        else:
            r1 = pc + (range_val * 1.1 / 12)
            s1 = pc - (range_val * 1.1 / 12)
        
        # Volume filter: current volume > 1.5x average
        vol_filter = volume[i] > 1.5 * avg_vol[i]
        
        if position == 0:
            # LONG: price breaks above R1, above 1d EMA34, and volume confirmation
            if close[i] > r1 and close[i] > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1, below 1d EMA34, and volume confirmation
            elif close[i] < s1 and close[i] < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price breaks below S1 OR trend fails (price below EMA)
            if close[i] < s1 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 OR trend fails (price above EMA)
            if close[i] > r1 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
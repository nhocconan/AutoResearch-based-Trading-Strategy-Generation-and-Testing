#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1D_Volume_Trend
Hypothesis: Camarilla R1/S1 levels from daily act as institutional support/resistance. 
Breaks with volume confirmation and daily trend filter capture strong moves. 
In bull markets: breaks above R1 in uptrend. 
In bear markets: breaks below S1 in downtrend. 
Volume filter avoids false breakouts. Target 25-35 trades/year.
"""

name = "4H_Camarilla_R1_S1_Breakout_1D_Volume_Trend"
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's range)
    camarilla_R1 = np.zeros_like(close_1d)
    camarilla_S1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_R1[i] = prev_close + range_val * 1.1 / 12
        camarilla_S1[i] = prev_close - range_val * 1.1 / 12
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).values
    
    # Align 1d indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + daily uptrend (close > EMA34)
            if (close[i] > camarilla_R1_aligned[i] and vol_spike and 
                close_1d[-1] > ema34_1d_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S1 + volume spike + daily downtrend (close < EMA34)
            elif (close[i] < camarilla_S1_aligned[i] and vol_spike and 
                  close_1d[-1] < ema34_1d_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or loss of volume/daily trend
            if (close[i] < camarilla_S1_aligned[i] or not vol_spike or 
                close_1d[-1] < ema34_1d_aligned[i] if len(close_1d) > 0 else True):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or loss of volume/daily trend
            if (close[i] > camarilla_R1_aligned[i] or not vol_spike or 
                close_1d[-1] > ema34_1d_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
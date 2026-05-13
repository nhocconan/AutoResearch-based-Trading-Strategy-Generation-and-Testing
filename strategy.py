#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_1dTrend_Volume
Hypothesis: Camarilla pivot reversals on 12h timeframe work in both bull and bear markets.
Buy near S1/S2 when 1d trend is up and volume confirms; sell near R1/R2 when 1d trend is down and volume confirms.
Uses 1d trend filter to avoid counter-trend trades in strong trends. Target: 15-30 trades/year per symbol.
"""

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla multipliers
    H_L = prev_high - prev_low
    R1 = prev_close + H_L * 1.1 / 12
    R2 = prev_close + H_L * 1.1 / 6
    R3 = prev_close + H_L * 1.1 / 4
    S1 = prev_close - H_L * 1.1 / 12
    S2 = prev_close - H_L * 1.1 / 6
    S3 = prev_close - H_L * 1.1 / 4
    
    # Align levels to 12h timeframe (they change only at 1d boundaries)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = prev_close > ema_50_1d
    downtrend_1d = prev_close < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 24-period average (24*12h = 12 days)
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any required data is not available
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Get current values
        r1 = R1_aligned[i]
        r2 = R2_aligned[i]
        s1 = S1_aligned[i]
        s2 = S2_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price near S1 or S2, 1d uptrend, volume confirmation
            if ((abs(close[i] - s1) / s1 < 0.005 or abs(close[i] - s2) / s2 < 0.005) and 
                uptrend and vol_conf):
                signals[i] = 0.25
                position = 1
            # SHORT: price near R1 or R2, 1d downtrend, volume confirmation
            elif ((abs(close[i] - r1) / r1 < 0.005 or abs(close[i] - r2) / r2 < 0.005) and 
                  downtrend and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches midpoint or 1d trend turns down
            midpoint = (s1 + r1) / 2  # Simple midpoint between S1 and R1
            if close[i] >= midpoint or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches midpoint or 1d trend turns up
            midpoint = (s1 + r1) / 2
            if close[i] <= midpoint or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
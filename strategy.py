#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use daily Camarilla pivot levels (R1/S1) as support/resistance for 12h breakouts.
Long when price breaks above R1 with volume confirmation; short when breaks below S1 with volume.
Only trade when 1w trend aligns (price above/below 1w EMA20) to avoid counter-trend whipsaws.
Targets 15-25 trades/year by requiring: Camarilla level break, volume > 1.5x 20-period average,
and 1w trend filter. Works in bull by buying R1 breaks in uptrend, in bear by selling S1 breaks
in downtrend. Volume ensures breakout legitimacy; trend filter reduces false signals.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) / 12.0
    r1_1d = close_1d + 1.1 * camarilla_range
    s1_1d = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for bar close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1, with volume, and uptrend (close > 1w EMA20)
            if (close[i] > r1_12h[i] and vol_confirm[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, with volume, and downtrend (close < 1w EMA20)
            elif (close[i] < s1_12h[i] and vol_confirm[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below R1 or trend changes (close < 1w EMA20)
            if (close[i] < r1_12h[i] or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or trend changes (close > 1w EMA20)
            if (close[i] > s1_12h[i] or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0
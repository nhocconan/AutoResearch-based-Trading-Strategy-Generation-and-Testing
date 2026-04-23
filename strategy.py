#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot (R1/S1) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses the Camarilla pivot point (PP).
Uses discrete position sizing (0.30) to minimize fee churn. Targets 20-50 trades/year per symbol.
The weekly EMA50 provides a robust trend filter that works in both bull and bear markets by avoiding counter-trend entries.
Volume confirmation filter set at 1.5x to reduce false breakouts while maintaining sufficient trade frequency.
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
    
    # Load 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on 1d data
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    # Range = High - Low
    range_ = high_1d - low_1d
    # Resistance levels
    r1 = pp + (range_ * 1.1 / 12)
    r2 = pp + (range_ * 1.1 / 6)
    r3 = pp + (range_ * 1.1 / 4)
    r4 = pp + (range_ * 1.1 / 2)
    # Support levels
    s1 = pp - (range_ * 1.1 / 12)
    s2 = pp - (range_ * 1.1 / 6)
    s3 = pp - (range_ * 1.1 / 4)
    s4 = pp - (range_ * 1.1 / 2)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Camarilla levels and 1w EMA50 to 1d timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND close > 1w EMA50 AND volume spike
            if (price > r1_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S1 AND close < 1w EMA50 AND volume spike
            elif (price < s1_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Camarilla PP
                if price < pp_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Camarilla PP
                if price > pp_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Camarilla_R1S1_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0
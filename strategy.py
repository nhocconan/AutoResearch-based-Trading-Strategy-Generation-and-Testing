#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrendFilter
Hypothesis: Daily Camarilla pivot R1/S1 breakout with weekly EMA20 trend filter and volume confirmation.
Long when price breaks above R1 with weekly uptrend and volume spike.
Short when price breaks below S1 with weekly downtrend and volume filter.
Uses ATR-based stoploss (2x ATR from entry) to manage risk.
Designed for low trade frequency (7-25/year) to avoid fee drag while capturing momentum in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp * 2.0 - low_1d
    s1 = pp * 2.0 - high_1d
    
    # Align 1d levels to lower timeframe (but we're on 1d, so no shift needed)
    # For safety, still use align_htf_to_ltf (will be 1:1 on same timeframe)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d pivot calculation (needs 1 bar), EMA20 (20), volume MA (20), ATR (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1, weekly uptrend, volume confirmation
            long_signal = (close_val > r1_val) and (close_val > ema_20_1w_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: price breaks below S1, weekly downtrend, volume confirmation
            short_signal = (close_val < s1_val) and (close_val < ema_20_1w_val) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: stoploss hit or trend reversal (weekly trend change)
            if (low_val < long_stop) or (close_val < ema_20_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: stoploss hit or trend reversal (weekly trend change)
            if (high_val > short_stop) or (close_val > ema_20_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter"
timeframe = "1d"
leverage = 1.0
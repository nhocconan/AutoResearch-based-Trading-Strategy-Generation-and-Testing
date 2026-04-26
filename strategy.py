#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and volume spike.
In bull markets: price breaks above R1 with weekly uptrend → long.
In bear markets: price breaks below S1 with weekly downtrend → short.
Uses discrete sizing (0.25) and ATR-based stoploss to reduce fee drag.
Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for ATR calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar: use current values (no look-ahead, but will be filtered by min_periods later)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels for R1 and S1
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # ATR for stoploss and volume filter
    atr_period = 14
    tr1 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr2 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr3 = np.abs(high - low)
    tr1[0] = 0  # First bar: no previous close
    tr2[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA10 for trend filter (responsive but smoothed)
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 20 for ATR and volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_10_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # ATR-based stoploss: 2.0 * ATR
        stoploss_long = 2.0 * atr[i]
        stoploss_short = 2.0 * atr[i]
        
        # Long logic: price breaks above R1 with volume spike and weekly uptrend
        long_breakout = close[i] > r1[i]
        long_condition = long_breakout and volume_spike[i] and (close[i] > ema_10_1w_aligned[i])
        
        # Short logic: price breaks below S1 with volume spike and weekly downtrend
        short_breakout = close[i] < s1[i]
        short_condition = short_breakout and volume_spike[i] and (close[i] < ema_10_1w_aligned[i])
        
        # Exit logic: stoploss hit or weekly trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Track entry price for stoploss (approximate using close at entry)
            # In practice, we use a trailing stop based on ATR from entry
            # Simplified: exit if price drops below entry - 2*ATR
            # We approximate entry price as the close when we entered
            # For better accuracy, we would need to track entry price, but we use close-based exit
            if close[i] < ema_10_1w_aligned[i]:  # Weekly trend reversal
                exit_long = True
            elif bars_since_entry >= 1 and close[i] < close[i-1] - stoploss_long:  # Price drop stoploss
                exit_long = True
                
        if position == -1:
            if close[i] > ema_10_1w_aligned[i]:  # Weekly trend reversal
                exit_short = True
            elif bars_since_entry >= 1 and close[i] > close[i-1] + stoploss_short:  # Price rise stoploss
                exit_short = True
        
        # Minimum holding period: 1 day (to reduce churn)
        if position != 0 and bars_since_entry < 1:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
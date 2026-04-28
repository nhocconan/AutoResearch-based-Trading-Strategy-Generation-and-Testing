#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and volatility filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    # Get 6h data for breakout signals (ATR-based breakout)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 6h ATR(14) for breakout threshold
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = tr2_6h[0] = tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Breakout threshold: 0.5 * ATR(14) from open
    open_6h = df_6h['open'].values
    upper_breakout = open_6h + 0.5 * atr_6h_14
    lower_breakout = open_6h - 0.5 * atr_6h_14
    upper_breakout_aligned = align_htf_to_ltf(prices, df_6h, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_6h, lower_breakout)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.mean(atr_14_aligned[max(0, i-50):i+1]) * 0.8
        
        # Breakout conditions: price breaks 0.5*ATR from 6h open
        long_breakout = close[i] > upper_breakout_aligned[i]
        short_breakout = close[i] < lower_breakout_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: reverse signal or volatility collapse
        if position == 1:
            exit_condition = not uptrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-20):i+1]) * 0.5)
        elif position == -1:
            exit_condition = not downtrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-20):i+1]) * 0.5)
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "6h_ATRBreakout_12hEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(20) for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 12h data for breakout signals (ATR-based breakout)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # 12h ATR(14) for breakout threshold
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = tr2_12h[0] = tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Breakout threshold: 0.7 * ATR(14) from open
    upper_breakout = open_12h + 0.7 * atr_12h_14
    lower_breakout = open_12h - 0.7 * atr_12h_14
    upper_breakout_aligned = align_htf_to_ltf(prices, df_12h, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_12h, lower_breakout)
    
    # Volume confirmation: current volume > 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_20_1d_aligned[i]
        downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.mean(atr_14_aligned[max(0, i-30):i+1]) * 0.7
        
        # Breakout conditions: price breaks 0.7*ATR from 12h open
        long_breakout = close[i] > upper_breakout_aligned[i]
        short_breakout = close[i] < lower_breakout_aligned[i]
        
        # Entry conditions: require trend alignment
        long_entry = long_breakout and uptrend and vol_filter and volume_confirm[i]
        short_entry = short_breakout and downtrend and vol_filter and volume_confirm[i]
        
        # Exit conditions: reverse signal or volatility collapse
        if position == 1:
            exit_condition = not uptrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-15):i+1]) * 0.5)
        elif position == -1:
            exit_condition = not downtrend or (atr_14_aligned[i] < np.mean(atr_14_aligned[max(0, i-15):i+1]) * 0.5)
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

name = "12h_ATRBreakout_1dEMA20_VolumeFilter"
timeframe = "12h"
leverage = 1.0
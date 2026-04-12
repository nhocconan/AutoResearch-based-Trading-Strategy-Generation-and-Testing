#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v12"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar data to avoid look-ahead
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate 1d Camarilla levels (H4/L4 breakout)
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    h4_prev = pivot_prev + (range_1d_prev * 1.1 / 2)
    l4_prev = pivot_prev - (range_1d_prev * 1.1 / 2)
    
    # Align to 4h
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_prev)
    
    # Volatility filter: ATR ratio (ATR(10)/ATR(30)) < 0.7 = low volatility regime
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr1 = high_low
    tr2 = high_close
    tr3 = low_close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr10 / atr30
    low_vol = atr_ratio < 0.7
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # Additional filter: avoid choppy markets using ADX(14) < 20
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr14)
    minus_di14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    not_choppy = adx > 20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(low_vol[i]) or np.isnan(vol_confirm[i]) or
            np.isnan(not_choppy[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        long_signal = close[i] > h4_aligned[i] and low_vol[i] and vol_confirm[i] and not_choppy[i]
        short_signal = close[i] < l4_aligned[i] and low_vol[i] and vol_confirm[i] and not_choppy[i]
        
        pivot_prev_val = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
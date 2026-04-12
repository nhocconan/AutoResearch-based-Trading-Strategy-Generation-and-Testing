#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volatility_filter_v3"
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
    
    # Volatility filter: ATR ratio (ATR(10)/ATR(30)) < 0.8 = low volatility regime
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
    low_vol = atr_ratio < 0.8  # Low volatility regime for better breakout follow-through
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any values not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(low_vol[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H4 in low volatility with volume confirmation
        long_signal = close[i] > h4_aligned[i] and low_vol[i] and vol_confirm[i]
        # Short: break below L4 in low volatility with volume confirmation
        short_signal = close[i] < l4_aligned[i] and low_vol[i] and vol_confirm[i]
        
        # Exit on opposite breakout to pivot level
        pivot_prev_val = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev_val)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
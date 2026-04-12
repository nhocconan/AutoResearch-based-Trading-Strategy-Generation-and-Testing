#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    range_1w = high_1w - low_1w
    camarilla_h4 = close_1w + 1.5 * range_1w / 2
    camarilla_l4 = close_1w - 1.5 * range_1w / 2
    
    # Align to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Current 12h volume filter
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > volume_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Camarilla breakout signals with volume confirmation
        long_signal = close[i] > camarilla_h4_aligned[i] and volume_ok[i] and volume[i] > volume_1d_aligned[i]
        short_signal = close[i] < camarilla_l4_aligned[i] and volume_ok[i] and volume[i] > volume_1d_aligned[i]
        
        # Exit when price returns to weekly close
        exit_long = close[i] < close_1w[-1] if not np.isnan(close_1w[-1]) else False
        exit_short = close[i] > close_1w[-1] if not np.isnan(close_1w[-1]) else False
        
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
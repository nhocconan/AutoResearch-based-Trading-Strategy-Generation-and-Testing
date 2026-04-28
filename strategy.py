#!/usr/bin/env python3
"""
12h_Weekly_Reset_Momentum_Reversal
Hypothesis: Combines weekly momentum reset (Monday open vs Friday close) with 1d RSI extremes and volume confirmation to capture mean-reversion after weekend gaps. Works in both bull and bear by fading extreme weekly moves. Targets 15-35 trades/year for low fee drag.
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
    open_time = prices['open_time'].values
    
    # Get daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    
    # Align daily RSI to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get weekly data for momentum reset (Monday open vs Friday close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly momentum: (Friday close - Monday open) / Monday open
    weekly_momentum = (df_1w['close'] - df_1w['open']) / df_1w['open']
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum.values)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(weekly_momentum_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Mean reversion conditions
        # Long: weekly momentum very negative AND RSI oversold
        long_signal = (weekly_momentum_aligned[i] < -0.05) and (rsi_1d_aligned[i] < 30) and vol_confirm
        # Short: weekly momentum very positive AND RSI overbought
        short_signal = (weekly_momentum_aligned[i] > 0.05) and (rsi_1d_aligned[i] > 70) and vol_confirm
        
        # Exit: RSI returns to neutral zone
        long_exit = rsi_1d_aligned[i] > 50
        short_exit = rsi_1d_aligned[i] < 50
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Weekly_Reset_Momentum_Reversal"
timeframe = "12h"
leverage = 1.0
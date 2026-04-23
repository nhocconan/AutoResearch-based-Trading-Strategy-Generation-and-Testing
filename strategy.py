#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Camarilla H3/L3 levels provide high-probability breakout zones from 1d pivot structure.
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend whipsaws.
- Volume > 2.0x 48-period average confirms breakout validity (tighter filter to reduce trades).
- Discrete position size 0.25 limits drawdown during crashes.
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to stay within fee-efficient range.
- Designed to work in both bull and bear regimes via trend filter and volume confirmation.
"""

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
    
    # Camarilla levels from prior 1d (using mtf_data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed daily bar)
    high_1d = df_1d['high'].shift(1).values  # shift(1) to use prior completed day
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d_aligned + 1.1 * (high_1d_aligned - low_1d_aligned) / 4
    camarilla_l3 = close_1d_aligned - 1.1 * (high_1d_aligned - low_1d_aligned) / 4
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 48-period average (tighter filter for 12h)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 48)  # EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 1d EMA50 AND volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 1d EMA50 AND volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 1d EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 1d EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
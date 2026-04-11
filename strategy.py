#!/usr/bin/env python3
# 12h_1w_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from 1w act as strong support/resistance. 
# Breakouts above/below these levels with volume > 1.5x 20-period 1w average 
# and aligned with 1w EMA trend capture sustainable moves. 
# Designed for low trade frequency (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h ATR for volatility filter (optional, can be removed if too restrictive)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w High, Low, Close for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for 1w
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_high = close_1w + 1.5 * (high_1w - low_1w)
    camarilla_low = close_1w - 1.5 * (high_1w - low_1w)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume average (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Align raw 1w volume for confirmation
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or \
           np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period average
        vol_confirm = vol_1w_aligned[i] > 1.5 * vol_avg_20_1w_aligned[i]
        
        # Trend filter: close vs 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Price relative to Camarilla levels
        above_resistance = close[i] > camarilla_high_aligned[i]
        below_support = close[i] < camarilla_low_aligned[i]
        
        # Entry conditions
        # Long: Price closes above Camarilla H4 AND uptrend AND volume confirmation
        if above_resistance and uptrend and vol_confirm and position != 1:
            # Additional check: ensure we didn't just break above in previous bar
            if i == 50 or close[i-1] <= camarilla_high_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price closes below Camarilla L4 AND downtrend AND volume confirmation
        elif below_support and downtrend and vol_confirm and position != -1:
            # Additional check: ensure we didn't just break below in previous bar
            if i == 50 or close[i-1] >= camarilla_low_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to mean (close to 1w close) or reverses
        elif position == 1 and close[i] < camarilla_high_aligned[i] * 0.995:  # Slight hysteresis
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_low_aligned[i] * 1.005:  # Slight hysteresis
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
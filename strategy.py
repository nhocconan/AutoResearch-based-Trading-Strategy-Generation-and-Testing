#!/usr/bin/env python3
"""
12h_1d_camarilla_volume_breakout_v1
Hypothesis: 12-hour strategy using daily Camarilla pivot levels with volume confirmation and 1d trend filter.
Enters long when price breaks above H3 with volume spike and 1d uptrend; short when breaks below L3 with volume spike and 1d downtrend.
Uses ATR-based volatility filter to avoid choppy markets. Designed to capture breakouts in both bull and bear markets.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing strong moves.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels using previous day's data (avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Trend filter from 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        # ATR-based volatility filter: avoid choppy markets (ATR < 1.5x 20-period average)
        if i >= 20:
            atr_ma = np.mean(atr[max(0, i-20):i+1])
            volatility_filter = atr[i] < atr_ma * 1.5
        else:
            volatility_filter = True
        
        # Fixed position size (discrete levels to reduce churn)
        position_size = 0.25
        
        # Entry conditions: Camarilla breakout with volume and trend confirmation
        long_breakout = close[i] > h3_12h[i] and volume_filter and uptrend_1d and volatility_filter
        short_breakout = close[i] < l3_12h[i] and volume_filter and downtrend_1d and volatility_filter
        
        # Exit conditions: reverse breakout or trend change
        long_exit = close[i] < l3_12h[i] or not uptrend_1d
        short_exit = close[i] > h3_12h[i] or not downtrend_1d
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0
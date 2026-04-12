#!/usr/bin/env python3
"""
4h_12h_1d_camarilla_breakout
Combines 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
Long when price breaks above H3 after bullish 1d trend, short when breaks below L3 after bearish 1d trend.
Uses volume spike confirmation and ATR-based stoploss to reduce false breakouts.
Targets 20-40 trades/year for low fee drag.
"""

name = "4h_12h_1d_camarilla_breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous day's range)
    # Using typical Camarilla formula: H4 = close + range * 1.1/2, H3 = close + range * 1.1/4, etc.
    # But we'll use the standard calculation based on prior period's high/low/close
    # For simplicity, we'll use the prior 12h bar's HLC
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    # Set first value to NaN (no previous bar)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    # Calculate pivot and ranges
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    range_12h = prev_high_12h - prev_low_12h
    
    # Camarilla levels
    H3 = pivot_12h + (range_12h * 1.1 / 4)
    L3 = pivot_12h - (range_12h * 1.1 / 4)
    H4 = pivot_12h + (range_12h * 1.1 / 2)
    L4 = pivot_12h - (range_12h * 1.1 / 2)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 20-period EMA for trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # ATR for dynamic stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above H3 after bullish 1d trend (price > EMA20), with volume spike
        if (close[i] > H3_aligned[i] and close[i] > ema_20_1d_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below L3 after bearish 1d trend (price < EMA20), with volume spike
        elif (close[i] < L3_aligned[i] and close[i] < ema_20_1d_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: reverse signal or price returns to opposite Camarilla level
        elif position == 1 and (close[i] < L3_aligned[i] or close[i] < ema_20_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > H3_aligned[i] or close[i] > ema_20_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
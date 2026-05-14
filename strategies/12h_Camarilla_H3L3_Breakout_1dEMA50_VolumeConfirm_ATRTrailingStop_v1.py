#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 level AND close > 1d EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 level AND close < 1d EMA50 (bearish trend)
- Volume must be > 2.0x 20-period average for confirmation (tight filter to reduce trades)
- ATR(14) trailing stop: exit when price moves 2.0x ATR from extreme since entry
- Uses 12h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla H3/L3 levels provide strong intraday support/resistance
- EMA50 trend filter prevents counter-trend entries during strong moves
- Volume confirmation ensures breakout legitimacy
- Designed to work in both bull and bear markets by following the 1d trend
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
    
    # Calculate Camarilla levels for current 12h bar using previous bar's range
    # Camarilla H3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    # Camarilla L3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    camarilla_h3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    camarilla_l3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average volume (tight filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    # ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price drops 2.0x ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.0x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_VolumeConfirm_ATRTrailingStop_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_volatility_breakout_v1
# Uses weekly Bollinger Band breakouts with daily ATR filter and volume confirmation.
# Works in bull markets by capturing breakouts above upper BB, and in bear markets
# by shorting breakdowns below lower BB. Volatility filter ensures trades occur during
# high-momentum periods, reducing false signals. Target: 15-25 trades/year per symbol.
name = "12h_1w_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    bb_mid = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Align BB to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 1.3 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 20-period average
        atr_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean()
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma.values)
        vol_filter = atr_1d_aligned[i] > atr_ma_aligned[i] if not np.isnan(atr_ma_aligned[i]) else False
        
        # Skip if volatility or volume filter fails
        if not (vol_filter and vol_confirm[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly upper BB
        if close[i] > bb_upper_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly lower BB
        elif close[i] < bb_lower_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < bb_lower_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > bb_upper_aligned[i] and position == -1:
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
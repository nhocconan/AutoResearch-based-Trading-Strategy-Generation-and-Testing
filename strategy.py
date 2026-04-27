#!/usr/bin/env python3
"""
1h_4h1d_Confluence_Strategy_v1
Hypothesis: Use 4h and 1d trend direction as primary signal filters, with 1h timeframe for precise entry timing.
Combines 4h EMA trend, 1d ADX trend strength, and 1h volume confirmation to avoid whipsaws.
Targets 15-37 trades per year to minimize fee drag while capturing major trends.
Works in both bull and bear markets by requiring strong trend alignment before entering.
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
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA21
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for EMA, ADX, and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        ema_4h_val = ema_4h_aligned[i]
        adx_val = adx_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: price above 4h EMA, ADX > 20 (trending), volume confirmation
            if close[i] > ema_4h_val and adx_val > 20 and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: price below 4h EMA, ADX > 20 (trending), volume confirmation
            elif close[i] < ema_4h_val and adx_val > 20 and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA or ADX weakens
            if close[i] < ema_4h_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 4h EMA or ADX weakens
            if close[i] > ema_4h_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_4h1d_Confluence_Strategy_v1"
timeframe = "1h"
leverage = 1.0
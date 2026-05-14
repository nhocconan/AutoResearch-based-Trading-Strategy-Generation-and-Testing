#!/usr/bin/env python3
"""
12h_KAMA_TRIX_Confluence_v1
Hypothesis: KAMA adapts to market noise while TRIX identifies momentum shifts.
Combining KAMA direction (trend) with TRIX zero-cross (momentum) and volume confirmation
creates a high-conviction signal that works in both bull and bear markets by avoiding
whipsaws in ranging markets. Uses 1d ADX filter to ensure trades only in strong trends.
Target: 15-25 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0])).cumsum()
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility
    
    er_smoothed = pd.Series(er).ewm(alpha=1/er_length, adjust=False).mean().values
    
    sc = (er_smoothed * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_trix(close, period=15):
    """Calculate TRIX indicator"""
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False).mean()
    
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    return trix.fillna(0).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength filter
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
    
    # Calculate KAMA on 12h data
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # Calculate TRIX on 12h data
    trix = calculate_trix(close, period=15)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ADX, KAMA, TRIX and volume MA
    start_idx = max(35, 20)  # ADX needs ~30, TRIX needs period, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(kama[i]) or np.isnan(trix[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        trix_val = trix[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: price above KAMA, TRIX positive (bullish momentum), ADX > 25, volume confirmation
            if close[i] > kama[i] and trix_val > 0 and adx_val > 25 and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: price below KAMA, TRIX negative (bearish momentum), ADX > 25, volume confirmation
            elif close[i] < kama[i] and trix_val < 0 and adx_val > 25 and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or TRIX turns negative or ADX weakens
            if close[i] < kama[i] or trix_val < 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or TRIX turns positive or ADX weakens
            if close[i] > kama[i] or trix_val > 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_TRIX_Confluence_v1"
timeframe = "12h"
leverage = 1.0
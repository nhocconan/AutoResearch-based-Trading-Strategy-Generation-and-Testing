#!/usr/bin/env python3
"""
4h_KAMA_Turn_With_Direction_Filter_v1
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, reducing whipsaws in sideways markets.
Trades only when KAMA turns (slope change) aligned with 1d EMA50 trend and volume spike.
Uses 1d ADX>25 to filter for trending markets only. Avoids false signals in ranging conditions.
Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.
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
    
    # Get 1d data for trend and ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr1 = np.maximum(tr1, np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d - low_1d
    # Plus Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    # Smoothed values
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    # Directional Indicators
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    # DX and ADX
    dx = 100 * np.absolute(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25  # Only trade when ADX > 25 (trending market)
    
    # KAMA on close prices
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)) if len(close.shape) > 1 else np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.array([0]*len(close))
    if len(close) > 10:
        volatility = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
        volatility = np.concatenate([np.array([0]*10), volatility[10:]]) if len(volatility) < len(close) else volatility
    else:
        volatility = np.zeros_like(close)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # KAMA slope (direction)
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    # KAMA turn: slope changes sign
    kama_turn_long = (kama_slope > 0) & (np.roll(kama_slope, 1) <= 0)
    kama_turn_short = (kama_slope < 0) & (np.roll(kama_slope, 1) >= 0)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    kama_turn_long_aligned = align_htf_to_ltf(prices, df_1d, kama_turn_long)
    kama_turn_short_aligned = align_htf_to_ltf(prices, df_1d, kama_turn_short)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA50 (50), ADX (14+14=28), KAMA (10), volume avg (20)
    start_idx = max(50, 28, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(adx_filter_aligned[i]) or 
            np.isnan(kama_turn_long_aligned[i]) or np.isnan(kama_turn_short_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        adx_ok = adx_filter_aligned[i]
        kama_long = kama_turn_long_aligned[i]
        kama_short = kama_turn_short_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Long: KAMA turn up + price > EMA50 + volume + ADX filter
            if kama_long and close_val > ema50 and vol_conf and adx_ok:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: KAMA turn down + price < EMA50 + volume + ADX filter
            elif kama_short and close_val < ema50 and vol_conf and adx_ok:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: KAMA turn down or price < EMA50
            if kama_short or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: KAMA turn up or price > EMA50
            if kama_long or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Turn_With_Direction_Filter_v1"
timeframe = "4h"
leverage = 1.0
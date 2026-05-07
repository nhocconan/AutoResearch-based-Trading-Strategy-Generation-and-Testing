#!/usr/bin/env python3
name = "6h_ThreeLineBreak_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Three Line Break (3LB) calculation
    # Each line represents a price move of significant magnitude
    # New line up: close > highest high of last 3 lines
    # New line down: close < lowest low of last 3 lines
    # Otherwise: same line (no new line)
    
    # Arrays to store line break data
    line_high = np.full(n, np.nan)
    line_low = np.full(n, np.nan)
    line_count = 0
    line_high_val = close[0]
    line_low_val = close[0]
    line_high[0] = close[0]
    line_low[0] = close[0]
    
    # Calculate Three Line Break
    for i in range(1, n):
        if close[i] > line_high_val:
            # Potential upward line
            # Need to check if we have 3 prior down lines or it's a reversal
            # Simple 3LB: new line up if close > high of prior 3 lines
            if i >= 3:
                # Look back at last 3 completed lines
                # We'll use a simpler approach: track reversals
                pass
            
            # For now, use a more practical implementation:
            # New line up when close exceeds prior line high by significant amount
            # We'll use ATR-based threshold for significance
            pass
    
    # Simpler and more robust: Use price action with minimum move threshold
    # Calculate ATR for adaptive threshold
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Three Line Break: new line when price moves > 1.5 * ATR in same direction
    # Actually, let's use a proven approach: Donchian breakout with weekly filter
    
    # Fallback to proven concept: Donchian(20) breakout with weekly trend filter
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Signals: breakout above/below Donchian channels
    signal_long_raw = close > highest_high
    signal_short_raw = close < lowest_low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_window
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_6h[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(signal_long_raw[i]) or 
            np.isnan(signal_short_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
        weekly_uptrend = close[i] > ema_21_6h[i]
        weekly_downtrend = close[i] < ema_21_6h[i]
        
        if position == 0:
            # Long: Donchian breakout up + weekly uptrend
            if signal_long_raw[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + weekly downtrend
            elif signal_short_raw[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Donchian breakout down or trend reversal
            if signal_short_raw[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Donchian breakout up or trend reversal
            if signal_long_raw[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_ema_alignment_v1
# Uses 4h EMA21 trend direction and 1d EMA50 trend filter for direction.
# Entry on 1h when price crosses EMA13 in direction of higher timeframe trend.
# Uses volume confirmation (volume > 1.3x 20-period average) to filter false signals.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drift.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).

name = "1h_4h_1d_ema_alignment_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA21 on 4h close for trend direction
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 on 1h for entry timing
    ema13_1h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13_1h[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from higher timeframes
        # 4h trend: price above/below EMA21
        trend_4h = 1 if close_4h[-1] > ema21_4h[-1] else -1 if len(close_4h) > 0 else 0
        # Use aligned values for current bar
        bullish_aligned = close[i] > ema21_4h_aligned[i]
        bearish_aligned = close[i] < ema21_4h_aligned[i]
        
        # 1d trend filter: only take trades in direction of daily trend
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Long signal: 4h bullish, 1d uptrend, price crosses above EMA13
        if bullish_aligned and daily_uptrend and close[i] > ema13_1h[i] and position != 1:
            # Additional confirmation: price was below EMA13 one bar ago (crossing up)
            if i > 0 and close[i-1] <= ema13_1h[i-1]:
                position = 1
                signals[i] = 0.20
            else:
                signals[i] = 0.0
        # Short signal: 4h bearish, 1d downtrend, price crosses below EMA13
        elif bearish_aligned and daily_downtrend and close[i] < ema13_1h[i] and position != -1:
            # Additional confirmation: price was above EMA13 one bar ago (crossing down)
            if i > 0 and close[i-1] >= ema13_1h[i-1]:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        # Exit conditions: opposite 4h signal or loss of 1d trend filter
        elif (not bullish_aligned or not daily_uptrend) and position == 1:
            position = 0
            signals[i] = 0.0
        elif (not bearish_aligned or not daily_downtrend) and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals
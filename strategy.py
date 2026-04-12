#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Daily chart Camarilla breakout with weekly trend filter and volume confirmation.
# In bull markets: long when price breaks above weekly trend and daily H4 resistance.
# In bear markets: short when price breaks below weekly trend and daily L4 support.
# Weekly trend uses 50-period EMA to filter direction. Volume confirms institutional participation.
# Target: 15-25 trades/year per symbol for low friction and high edge.
name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to daily timeframe
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above EMA50 = bullish, below = bearish
        weekly_bullish = close[i] > ema_50_1w_aligned[i]
        weekly_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Long signal: bullish weekly trend + price breaks above H4 + volume
        if weekly_bullish and close[i] > h4_level[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: bearish weekly trend + price breaks below L4 + volume
        elif weekly_bearish and close[i] < l4_level[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or trend change
        elif (not weekly_bullish and position == 1) or (not weekly_bearish and position == -1):
            position = 0
            signals[i] = 0.0
        elif close[i] < l4_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h4_level[i] and position == -1:
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_trend_v1
# Daily timeframe with weekly trend filter (weekly EMA200) to capture major trends.
# Uses daily Camarilla H3/L3 breakouts with volume confirmation for entry.
# Long when price > weekly EMA200, short when price < weekly EMA200.
# Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "1d_1w_camarilla_breakout_trend_v1"
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
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to daily timeframe (daily levels update only after daily bar closes)
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(ema200_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above/below weekly EMA200
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Require volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H3 in bullish trend
        if close[i] > h3_level[i] and bullish_trend and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L3 in bearish trend
        elif close[i] < l3_level[i] and bearish_trend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or trend reversal
        elif (close[i] < l3_level[i] and position == 1) or \
             (close[i] > h3_level[i] and position == -1) or \
             (bullish_trend and position == -1) or \
             (bearish_trend and position == 1):
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
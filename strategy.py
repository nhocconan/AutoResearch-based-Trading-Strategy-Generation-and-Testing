#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_camarilla_breakout_volume_v1
# Uses daily Camarilla levels to determine intraday 6h bias.
# Long when price breaks above daily H3 with volume confirmation and 12h trend up (EMA50 > EMA200).
# Short when price breaks below daily L3 with volume confirmation and 12h trend down (EMA50 < EMA200).
# Uses 12h EMA crossover for trend filter to avoid counter-trend trades.
# Designed for moderate trade frequency (target: 50-150 total trades over 4 years) to balance edge and cost.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "6h_12h_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and 12h data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (daily levels update only after daily bar closes)
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h EMA50 and EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_12h, ema200)
    
    # Volume confirmation: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # start after warmup
        # Skip if levels not ready
        if (np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine 12h trend
        trend_up = ema50_aligned[i] > ema200_aligned[i]
        trend_down = ema50_aligned[i] < ema200_aligned[i]
        
        # Long signal: price breaks above daily H3 with volume and 12h trend up
        if close[i] > h3_level[i] and trend_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L3 with volume and 12h trend down
        elif close[i] < l3_level[i] and trend_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
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
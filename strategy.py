#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v2
# Strategy: 4h Camarilla pivot breakout with daily trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Breakouts above/below these levels with volume confirmation and daily trend alignment
# capture momentum moves. Designed for moderate trade frequency (~30-50/year) to balance
# signal quality and fee drag in BTC/ETH.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # Pivot = (high + low + close) / 3
    # We'll use H3/L3 as breakout levels (stronger than H4/L4)
    
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.125 * range_1d
    camarilla_l3 = close_1d - 1.125 * range_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Volume confirmation: current volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions using H3/L3 levels
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Strong breakout conditions using H4/L4 levels (optional filter)
        strong_breakout_up = close[i] > camarilla_h4_aligned[i]
        strong_breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        # Require strong breakout (H4/L4) for entry to reduce false signals
        if strong_breakout_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_breakout_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite H3/L3 breakout with volume (sooner exit)
        elif position == 1 and breakout_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
# 12h_1w_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot levels with volume confirmation and 1w EMA trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Price rejecting these levels with volume confirmation indicates reversal potential. The 1w EMA filter ensures trades align with the higher timeframe trend. This combination should work in both bull and bear markets by capturing reversals at key levels. Target: 12-37 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # We need daily high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We'll use H3/L3 for entry and H4/L4 for stop
    
    rng = prev_high - prev_low
    camarilla_h3 = prev_close + 1.0 * rng
    camarilla_l3 = prev_close - 1.0 * rng
    camarilla_h4 = prev_close + 1.5 * rng
    camarilla_l4 = prev_close - 1.5 * rng
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Camarilla H3/L3 touch/rejection with volume and trend alignment
        # Long: price crosses above L3 with volume in uptrend
        # Short: price crosses below H3 with volume in downtrend
        if i > 0:
            # Check for crossover of L3 (long entry)
            if (close[i-1] <= camarilla_l3_aligned[i-1] and close[i] > camarilla_l3_aligned[i] and 
                vol_confirm[i] and uptrend and position != 1):
                position = 1
                signals[i] = 0.25
            # Check for crossover of H3 (short entry)
            elif (close[i-1] >= camarilla_h3_aligned[i-1] and close[i] < camarilla_h3_aligned[i] and 
                  vol_confirm[i] and downtrend and position != -1):
                position = -1
                signals[i] = -0.25
            # Exit: price reaches opposite H3/L3 level
            elif position == 1 and close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            signals[i] = 0.0
    
    return signals
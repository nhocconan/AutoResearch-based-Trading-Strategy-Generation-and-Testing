#!/usr/bin/env python3
# 4h_12h_camarilla_volume_crossover_v1
# Strategy: 4-hour Camarilla pivot level crossover with 12-hour volume surge and momentum filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. A crossover above/below
# key levels (H3/L3) with 12h volume surge (>2x average) and aligned 12h momentum (close > open)
# captures institutional breakouts. Works in bull by catching breakouts, in bear by catching
# breakdowns with volume confirmation. Low-frequency, high-conviction trades avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_crossover_v1"
timeframe = "4h"
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
    open_price = prices['open'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day (using 1d data for pivot calculation)
    # Since we're on 4h timeframe, we use daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla formulas: 
    # H4 = C + (H-L) * 1.1/2
    # H3 = C + (H-L) * 1.1/4
    # L3 = C - (H-L) * 1.1/4
    # L4 = C - (H-L) * 1.1/2
    # Where C = (H+L+C)/3 (typical price)
    
    # Get previous day's OHLC (we need to shift by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot points
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + range_hl * 1.1 / 4.0
    l3 = pivot - range_hl * 1.1 / 4.0
    h4 = pivot + range_hl * 1.1 / 2.0
    l4 = pivot - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 12h volume average for surge detection
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_surge_12h = vol_12h > (2.0 * vol_avg_12h)  # Volume surge: >2x average
    vol_surge_aligned = align_htf_to_ltf(prices, df_12h, vol_surge_12h)
    
    # 12h momentum filter: close > open (bullish candle)
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    momentum_12h = close_12h > open_12h  # Bullish momentum
    momentum_aligned = align_htf_to_ltf(prices, df_12h, momentum_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_surge_aligned[i]) or np.isnan(momentum_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: price crosses above H3 with volume surge and bullish 12h momentum
        long_breakout = (close[i] > h3_aligned[i]) and vol_surge_aligned[i] and momentum_aligned[i]
        # Short: price crosses below L3 with volume surge and bearish 12h momentum
        short_breakout = (close[i] < l3_aligned[i]) and vol_surge_aligned[i] and (not momentum_aligned[i])
        
        # Exit conditions: reverse signal or price reaches opposite extreme level
        exit_long = position == 1 and (close[i] < l3_aligned[i] or 
                                      (close[i] > h3_aligned[i] and not momentum_aligned[i]))
        exit_short = position == -1 and (close[i] > h3_aligned[i] or 
                                        (close[i] < l3_aligned[i] and momentum_aligned[i]))
        
        # Trading logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
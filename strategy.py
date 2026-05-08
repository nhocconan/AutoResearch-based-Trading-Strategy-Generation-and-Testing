#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 (12h) AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 (12h) AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Camarilla H-L range (between H4 and L4).
# Uses Camarilla pivot levels from higher timeframe for structural support/resistance.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull/bear via trend filter and volatility-based entry.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (based on previous 12h bar)
    # H4 = Close + 1.1*(High - Low)/2
    # L4 = Close - 1.1*(High - Low)/2
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    # Note: Camarilla R3/S3 are same as H4/L4 in standard calculation
    range_12h = high_12h - low_12h
    camarilla_H4 = close_12h + 1.1 * range_12h / 2
    camarilla_L4 = close_12h - 1.1 * range_12h / 2
    camarilla_R3 = camarilla_H4  # R3 = H4
    camarilla_S3 = camarilla_L4  # S3 = L4
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L4)
    camarilla_R3_aligned = camarilla_H4_aligned
    camarilla_S3_aligned = camarilla_L4_aligned
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, 12h EMA50 rising, volume filter
            long_cond = (close[i] > camarilla_R3_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Camarilla S3, 12h EMA50 falling, volume filter
            short_cond = (close[i] < camarilla_S3_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla L4 (S3 equivalent)
            if close[i] < camarilla_L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla H4 (R3 equivalent)
            if close[i] > camarilla_H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
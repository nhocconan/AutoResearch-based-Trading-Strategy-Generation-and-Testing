#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d EMA50 Trend Filter
# Uses weekly Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) to capture strong moves.
# Long on break above R4 with 1d uptrend, short on break below S4 with 1d downtrend.
# Reversals at R3/S3 with volume spike to catch overextended moves.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
# Weekly pivot structure provides clean support/resistance that works in both bull and bear markets.

name = "6h_WeeklyCamarilla_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one weekly bar
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Based on previous week's high, low, close
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # Camarilla formula: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    width = whigh - wlow
    r4 = wclose + (width * 1.1 / 2)
    r3 = wclose + (width * 1.1 / 4)
    s3 = wclose - (width * 1.1 / 4)
    s4 = wclose - (width * 1.1 / 2)
    
    # Align weekly levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with 1d uptrend
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_50_aligned[i]):  # 1d uptrend
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with 1d downtrend
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_50_aligned[i]):  # 1d downtrend
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S3 with volume spike in uptrend
            elif (close[i] < s3_aligned[i] and 
                  close[i] > ema_50_aligned[i] and  # 1d uptrend
                  volume_spike[i] and
                  low[i] < s3_aligned[i]):  # Price probed S3 level
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R3 with volume spike in downtrend
            elif (close[i] > r3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i] and
                  high[i] > r3_aligned[i]):  # Price probed R3 level
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 (failure) or 1d trend turns down
            if (close[i] < r3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 (failure) or 1d trend turns up
            if (close[i] > s3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
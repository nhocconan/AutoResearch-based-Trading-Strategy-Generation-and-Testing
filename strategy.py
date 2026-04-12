#!/usr/bin/env python3
"""
4h_1d_Camilla_Pivot_Range_Extreme_v1
Hypothesis: In ranging markets, price tends to revert from extreme Camarilla pivot levels (S3/R3).
Long when price closes below S3 with bullish engulfing candle and RSI(14)<30.
Short when price closes above R3 with bearish engulfing candle and RSI(14)>70.
Exit when price returns to the daily pivot point.
Uses 1d Camarilla levels for structure and 4h RSI for entry timing.
Designed for low trade frequency (<25/year) by requiring extreme level touches and reversal confirmation.
Works in bull/bear via mean reversion at statistical extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camilla_Pivot_Range_Extreme_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # === DAILY CAMARILLA PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (S1-S4, R1-R4)
    r3_1d = close_1d + range_1d * 1.1 / 2
    s3_1d = close_1d - range_1d * 1.1 / 2
    pivot_point_1d = pivot_1d  # Use as exit level
    
    # === 4H RSI(14) FOR MOMENTUM CONFIRMATION ===
    if len(close) >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full(n, 50.0)
    
    # Align data to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_point_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_point_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_point_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Candlestick patterns for reversal confirmation
        bullish_engulfing = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                            (close[i] >= open_price[i-1]) and (open_price[i] <= close[i-1])
        bearish_engulfing = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                            (open_price[i] >= close[i-1]) and (close[i] <= open_price[i-1])
        
        # Entry conditions: extreme level rejection with reversal confirmation
        long_setup = (close[i] <= s3_1d_aligned[i]) and bullish_engulfing and (rsi[i] < 30)
        short_setup = (close[i] >= r3_1d_aligned[i]) and bearish_engulfing and (rsi[i] > 70)
        
        # Exit conditions: return to pivot point (mean reversion)
        exit_long = close[i] >= pivot_point_1d_aligned[i]
        exit_short = close[i] <= pivot_point_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
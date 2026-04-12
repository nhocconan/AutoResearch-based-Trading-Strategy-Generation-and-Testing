#!/usr/bin/env python3
"""
6h_1w_Camarilla_Pivot_Breakout_Trend_v1
Hypothesis: Weekly Camarilla pivot levels with breakout/continuation logic.
In trending markets (ADX > 25), price breaking above R4 or below S4 continues.
In ranging markets (ADX < 20), price reverts from R3/S3 levels.
Uses 6h timeframe for execution and weekly Camarilla levels for structure.
Designed for low trade frequency (12-37 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Camarilla_Pivot_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX on 6h data for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Calculate DX
        dx = np.zeros_like(high)
        di_sum = plus_dm_smooth + minus_dm_smooth
        dx[period:] = np.where(
            di_sum[period:] != 0,
            abs(plus_dm_smooth[period:] - minus_dm_smooth[period:]) / di_sum[period:] * 100,
            0
        )
        
        # Calculate ADX
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close)
    
    # Load weekly data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    range_val = df_1w['high'] - df_1w['low']
    
    # Camarilla levels
    r4 = typical_price + range_val * 1.1 / 2
    r3 = typical_price + range_val * 1.1 / 4
    r2 = typical_price + range_val * 1.1 / 6
    r1 = typical_price + range_val * 1.1 / 12
    s1 = typical_price - range_val * 1.1 / 12
    s2 = typical_price - range_val * 1.1 / 6
    s3 = typical_price - range_val * 1.1 / 4
    s4 = typical_price - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: trending vs ranging
        trending_market = adx[i] > 25
        ranging_market = adx[i] < 20
        
        # Breakout conditions (trending market)
        long_breakout = trending_market and close[i] > r4_aligned[i]
        short_breakout = trending_market and close[i] < s4_aligned[i]
        
        # Mean reversion conditions (ranging market)
        long_reversion = ranging_market and close[i] < s3_aligned[i]
        short_reversion = ranging_market and close[i] > r3_aligned[i]
        
        # Exit conditions
        long_exit = (close[i] < r3_aligned[i]) or (close[i] > s3_aligned[i])
        short_exit = (close[i] > s3_aligned[i]) or (close[i] < r3_aligned[i])
        
        # Priority: breakout > reversion > exit > hold
        if (long_breakout or long_reversion) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_breakout or short_reversion) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
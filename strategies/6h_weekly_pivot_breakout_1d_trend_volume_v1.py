#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation
Hypothesis: Price breaking above/below weekly pivot levels on 6h timeframe,
filtered by daily trend direction and volume spikes, captures strong momentum moves
while avoiding false breakouts. Weekly pivots provide strong support/resistance levels
that work in both bull (breakouts above weekly resistance) and bear (breakdowns below weekly support).
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d data for weekly pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_pivot = df_1d['close'].values
    
    # Weekly pivot points (using prior week's data)
    # Calculate weekly high/low/close from daily data
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d_pivot).rolling(window=5, min_periods=5).last().values
    
    # Pivot point calculation: (H + L + C) / 3
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Support and resistance levels
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot_point - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # ATR for volatility filter (14-period)
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: ATR > 20-period ATR mean (avoid choppy markets)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    # Volume filter: current volume > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or trend reverses
            if close[i] < s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or trend reverses
            if close[i] > r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price breaks above R1 + uptrend + volume spike + vol filter
            if (close[i] > r1_aligned[i] and 
                uptrend and 
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 + downtrend + volume spike + vol filter
            elif (close[i] < s1_aligned[i] and 
                  downtrend and 
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
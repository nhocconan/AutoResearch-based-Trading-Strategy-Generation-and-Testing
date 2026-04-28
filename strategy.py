#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout_1dTrend_Filter_v1
Hypothesis: Use weekly pivot levels for major support/resistance, with 1d trend filter and volatility-based range breakout.
In bull markets: buy pullbacks to weekly S1/S2 in uptrend. In bear markets: sell rallies to weekly R1/R2 in downtrend.
Weekly pivots provide institutional levels; 1d trend filters for direction; range breakout avoids chop.
Targets 15-30 trades/year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivots from previous week
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volatility filter: avoid choppy markets (use 6h ATR)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Normalize ATR by price for volatility regime
    atr_ratio = atr / np.maximum(close, 1e-10)
    # Low volatility threshold: avoid extremely choppy periods
    vol_ma_50 = pd.Series(atr_ratio).rolling(window=50, min_periods=50).mean().values
    low_vol_filter = atr_ratio < (vol_ma_50 * 1.5)  # Avoid high volatility spikes
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(low_vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: pullback to pivot levels in trend direction with low volatility filter
        long_entry = (close[i] >= s1_aligned[i] and close[i] <= pivot_aligned[i] and 
                     d1_uptrend[i] and low_vol_filter[i])
        short_entry = (close[i] <= r1_aligned[i] and close[i] >= pivot_aligned[i] and 
                      d1_downtrend[i] and low_vol_filter[i])
        
        # Exit when price reaches opposite pivot level or trend reverses
        long_exit = (close[i] >= r1_aligned[i]) or (not d1_uptrend[i])
        short_exit = (close[i] <= s1_aligned[i]) or (not d1_downtrend[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_RangeBreakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0
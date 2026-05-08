#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot points (P1, R1, S1)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_pivot + weekly_range
    weekly_s1 = weekly_pivot - weekly_range
    
    # Align weekly pivot levels to 6h
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_6h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r1 = close > weekly_r1_6h * 1.003
    price_below_s1 = close < weekly_s1_6h * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(trend_1d_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume spike and 1d uptrend
            long_cond = (price_above_r1[i] and vol_spike[i] and trend_1d_6h[i] > 0.5)
            
            # Short entry: price breaks below weekly S1 with volume spike and 1d downtrend
            short_cond = (price_below_s1[i] and vol_spike[i] and trend_1d_6h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below weekly R1 (mean reversion)
            if close[i] < weekly_r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above weekly S1 (mean reversion)
            if close[i] > weekly_s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot R1/S1 breakout on 6h timeframe with 1d EMA50 trend filter and volume confirmation.
# Weekly pivots provide stronger institutional levels than daily pivots, reducing false breakouts.
# Works in both bull (breakouts) and bear (reversals at S1/R1) markets due to mean-reversion exit logic.
# Target: 20-40 trades/year to stay within optimal frequency range for 6h timeframe.
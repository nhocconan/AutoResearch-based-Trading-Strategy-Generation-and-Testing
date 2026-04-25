#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike confirmation.
- Trend filter: price > 12h EMA50 = bullish, price < 12h EMA50 = bearish
- In bullish 12h trend: buy breakouts above R1, sell breakdowns below S1
- In bearish 12h trend: sell breakdowns below S1, buy breakouts above R1 (counter-trend fade)
- Volume confirmation: require volume > 1.8x 20-period average to reduce false signals
- Position size: 0.25 discrete levels to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) - proven Camarilla structure with tighter volume filter
- Works in both bull and bear: 12h EMA50 adapts to medium-term trend, volume filters noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    # Fix first values
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels - focus on R1/S1 for entries, R3/S3 for stop/reversal zones
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike confirmation: volume > 1.8x 20-period average (tighter than 1.5x)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend using EMA50
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for breakout setups with volume confirmation
            long_breakout = (close[i] > r1_aligned[i]) and volume_spike[i]
            short_breakout = (close[i] < s1_aligned[i]) and volume_spike[i]
            
            if htf_12h_bullish and long_breakout:
                # Bullish 12h trend: buy R1 breakout
                signals[i] = 0.25
                position = 1
            elif htf_12h_bearish and short_breakout:
                # Bearish 12h trend: sell S1 breakdown
                signals[i] = -0.25
                position = -1
            # In ranging markets or conflicting signals, stay flat to avoid whipsaw
            
        elif position == 1:
            # Long position: hold until exit conditions
            signals[i] = 0.25
            
            # Exit conditions:
            # 1. Price reaches S1 (mean reversion target)
            # 2. Price reaches R3 (extended move, consider taking profit)
            # 3. 12h trend turns bearish (trend change)
            if (close[i] < s1_aligned[i]) or (close[i] > r3_aligned[i]) or htf_12h_bearish:
                signals[i] = 0.0
                position = 0
                
        elif position == -1:
            # Short position: hold until exit conditions
            signals[i] = -0.25
            
            # Exit conditions:
            # 1. Price reaches R1 (mean reversion target)
            # 2. Price reaches S3 (extended move, consider taking profit)
            # 3. 12h trend turns bullish (trend change)
            if (close[i] > r1_aligned[i]) or (close[i] < s3_aligned[i]) or htf_12h_bullish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
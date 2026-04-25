#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 1h timeframe with volume spike confirmation and 4h EMA50 trend filter capture short-term institutional order flow aligned with 4h trend. Uses 4h for signal direction (trend) and 1h for precise entry timing. Session filter (08-20 UTC) reduces noise trades. Target: 15-30 trades/year per symbol to minimize fee drag while maintaining edge. Works in bull markets (breakouts continue with trend) and bear markets (mean reversion at R3/S3 with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour
    
    # 4h data for EMA50 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 trend filter
    ema_50_4h = calculate_ema(df_4h['close'].values, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for Camarilla pivots (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla levels: R3, S3 (tighter bands for precision)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_range * 0.25  # R3 level
    s3 = prev_close - camarilla_range * 0.25  # S3 level
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20) + Camarilla (2)
    start_idx = max(50, 20, 2)
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 4h EMA50 trend alignment
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_4h_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_4h_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R3 (failed breakout) or trend turns bearish
            if curr_close < r3_aligned[i] or curr_close < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price closes above S3 (failed breakout) or trend turns bullish
            if curr_close > s3_aligned[i] or curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0
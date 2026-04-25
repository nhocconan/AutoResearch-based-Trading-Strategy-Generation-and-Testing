#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily timeframe Camarilla R3/S3 breakouts with volume spike confirmation and weekly EMA50 trend filter capture institutional order flow. Works in bull markets by riding breakouts with the trend and in bear markets by fading false breaks at R3/S3 when aligned with weekly trend. Target: 10-25 trades/year to minimize fee drag on 1d timeframe.
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
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla levels: R3, S3
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_range * 0.55
    s3 = prev_close - camarilla_range * 0.55
    
    # Align Camarilla levels to 1d timeframe (already completed 1d bar)
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
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 1w EMA50 trend alignment
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_50_1w_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R3 (failed breakout) or trend turns bearish
            if curr_close < r3_aligned[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above S3 (failed breakout) or trend turns bullish
            if curr_close > s3_aligned[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
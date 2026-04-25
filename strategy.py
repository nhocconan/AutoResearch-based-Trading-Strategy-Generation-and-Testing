#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance derived from 1d data.
Breakouts above R3 or below S3 with volume confirmation and aligned 1w EMA34 trend capture
swing moves in both bull and bear markets. Uses weekly timeframe for trend filter to reduce
noise while maintaining alignment with 4h structure. Designed for low trade frequency (20-50/year)
with clear entry/exit rules to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels (R3, R2, R1, PP, S1, S2, S3)"""
    if len(high) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]))
    
    high_val = high[-1]
    low_val = low[-1]
    close_val = close[-1]
    
    pp = (high_val + low_val + close_val) / 3
    range_val = high_val - low_val
    
    r3 = close_val + range_val * 1.1 / 4
    r2 = close_val + range_val * 1.1 / 2
    r1 = close_val + range_val * 1.1 / 6
    pp_val = pp
    s1 = close_val - range_val * 1.1 / 6
    s2 = close_val - range_val * 1.1 / 2
    s3 = close_val - range_val * 1.1 / 4
    
    return (r3, r2, r1, pp_val, s1, s2, s3)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data (using previous day's OHLC)
    # We need to shift by 1 to use previous day's data for today's levels
    high_shifted = df_1d['high'].shift(1)
    low_shifted = df_1d['low'].shift(1)
    close_shifted = df_1d['close'].shift(1)
    
    # Calculate Camarilla for each bar using previous day's data
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        r3, _, _, _, _, _, s3 = calculate_camarilla(
            high_shifted.iloc[i], low_shifted.iloc[i], close_shifted.iloc[i]
        )
        camarilla_r3.iloc[i] = r3
        camarilla_s3.iloc[i] = s3
    
    camarilla_r3 = camarilla_r3.values
    camarilla_s3 = camarilla_s3.values
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and aligned data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > r3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S3 AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < s3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below S3 (strong support broken) OR price crosses below EMA (trend change)
            if (curr_close < s3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above R3 (strong resistance broken) OR price crosses above EMA (trend change)
            if (curr_close > r3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
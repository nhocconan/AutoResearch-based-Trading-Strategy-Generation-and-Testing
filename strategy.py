#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as significant intraday support/resistance.
Breakouts above R3 or below S3 with volume confirmation and aligned 1d EMA34 trend capture
strong momentum moves in both bull and bear markets. Uses 1d timeframe for trend filter
to reduce whipsaw while maintaining 4h structure for timely entries. Designed for moderate
trade frequency (20-50/year) with clear rules that avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data: based on previous 4h bar's range
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    # Using previous completed 4h bar to avoid look-ahead
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    camarilla_r3 = prev_close + (1.1 * (prev_high - prev_low) / 2)
    camarilla_s3 = prev_close - (1.1 * (prev_high - prev_low) / 2)
    
    # AlCamarilla levels to 4h timeframe (already aligned via get_htf_data)
    camarilla_r3_aligned = camarilla_r3  # Already from 4h data
    camarilla_s3_aligned = camarilla_s3  # Already from 4h data
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla calculation (requires previous bar), EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Camarilla R3 (resistance) AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > camarilla_r3_aligned[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Camarilla S3 (support) AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < camarilla_s3_aligned[i]) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below Camarilla S3 (support broken) OR price crosses below EMA (trend change)
            if (curr_close < camarilla_s3_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Camarilla R3 (resistance broken) OR price crosses above EMA (trend change)
            if (curr_close > camarilla_r3_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
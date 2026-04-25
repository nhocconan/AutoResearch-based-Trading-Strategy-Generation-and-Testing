#!/usr/bin/env python3
"""
1d Camarilla Pivot R1/S1 Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) from daily data act as intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and aligned weekly EMA34 trend capture
swing moves in both bull and bear markets. Uses 1d timeframe for entries and 1w for trend filter
to reduce noise and avoid overtrading. Designed for low trade frequency (7-25/year) with clear
entry/exit rules to work in ranging and trending markets.
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
    """Calculate Camarilla pivot levels"""
    if len(high) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Pivot point
    pivot = (high + low + close) / 3
    
    # Calculate ranges
    range_hl = high - low
    
    # Camarilla levels
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    r2 = close + range_hl * 1.1 / 6
    s2 = close - range_hl * 1.1 / 6
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    r4 = close + range_hl * 1.1 / 2
    s4 = close - range_hl * 1.1 / 2
    
    return (pivot, r1, r2, r3, r4, s1, s2, s3, s4)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data
    pivot_1d, r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align Camarilla levels to 1d timeframe (no shift needed as we're already on 1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla, EMA, volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > r1_1d_aligned[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S1 AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < s1_1d_aligned[i]) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below S1 (support broken) OR price crosses below EMA (trend change)
            if (curr_close < s1_1d_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above R1 (resistance broken) OR price crosses above EMA (trend change)
            if (curr_close > r1_1d_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
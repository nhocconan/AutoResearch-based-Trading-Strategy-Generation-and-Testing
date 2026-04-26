#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_HTFTrend_v1
Hypothesis: Use Camarilla R1/S1 from daily pivots for breakout entries with volume spike (>2.0x 20-period average) and 1-week EMA50 trend filter. Targets 20-50 trades/year on 4h timeframe by using tight Camarilla levels (R1/S1) for fewer, higher-quality breakouts. Works in bull markets via trend-following breakouts and in bear markets via volume-confirmed breakouts aligned with weekly trend. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels R1/S1 (based on previous 1d bar's range)
    # Camarilla R1 = close + 1.1*(high - low)/4
    # Camarilla S1 = close - 1.1*(high - low)/4
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to 1d if insufficient 1w data
        close_1w_series = pd.Series(df_1d['close'].values)
    else:
        close_1w_series = pd.Series(df_1w['close'].values)
    
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d) + 1w EMA50 + volume MA
    start_idx = max(2, 50, 20)  # 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + 1w uptrend
            long_breakout = (close[i] > camarilla_r1_aligned[i]) and \
                           (close[i-1] > camarilla_r1_aligned[i-1]) and \
                           (close[i-2] > camarilla_r1_aligned[i-2])
            long_signal = long_breakout and volume_spike[i] and trend_1w_uptrend
            
            # Short: price breaks below S1 + volume spike + 1w downtrend
            short_breakout = (close[i] < camarilla_s1_aligned[i]) and \
                           (close[i-1] < camarilla_s1_aligned[i-1]) and \
                           (close[i-2] < camarilla_s1_aligned[i-2])
            short_signal = short_breakout and volume_spike[i] and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S1 OR 1w trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR 1w trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_HTFTrend_v1"
timeframe = "4h"
leverage = 1.0
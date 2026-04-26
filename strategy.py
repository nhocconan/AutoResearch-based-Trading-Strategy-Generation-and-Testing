#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike filter.
R1/S1 levels provide tighter stops and more frequent valid breaks than R4/S4.
4h trend ensures we trade with the intermediate trend, reducing whipsaws.
1d volume spike (>1.5x 20-period average) confirms institutional interest.
Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Uses discrete position sizing (0.0, ±0.20) to reduce fee churn.
Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes with trend filter).
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
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 4.0)
    s1 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d volume spike filter: volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 4h EMA(50) and 1d volume MA(20)
    start_idx = 50 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        vol_filter = vol_spike_aligned[i]  # 1d volume spike
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend up AND volume spike
            long_signal = (close_val > r1_aligned[i]) and trend_4h_up and vol_filter
            
            # Short: price breaks below S1 AND 4h trend down AND volume spike
            short_signal = (close_val < s1_aligned[i]) and trend_4h_down and vol_filter
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend flips down OR volume spike ends
            if not trend_4h_up or not vol_filter:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR volume spike ends
            if not trend_4h_down or not vol_filter:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, use Camarilla R1/S1 breakout with 1w trend filter and volume confirmation for entries. Go long when price breaks above R1 with bullish 1w trend (close > 1w EMA34) and volume spike (>1.8x 20-period average). Go short when price breaks below S1 with bearish 1w trend (close < 1w EMA34) and volume spike. Exit when price reverts to the Camarilla pivot point (PP). Designed for 12-37 trades/year on 12h by requiring multi-timeframe alignment and volume confirmation, reducing fee drag while capturing strong trending moves in both bull and bear markets. The 1w trend filter ensures we only trade with the dominant weekly trend, improving win rate during bear markets like 2022 and 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Use the last completed 1d bar (index -1) to avoid look-ahead
    prev_close = df_1d['close'].iloc[-1].values if hasattr(df_1d['close'].iloc[-1], 'values') else df_1d['close'].iloc[-1]
    prev_high = df_1d['high'].iloc[-1].values if hasattr(df_1d['high'].iloc[-1], 'values') else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-1].values if hasattr(df_1d['low'].iloc[-1], 'values') else df_1d['low'].iloc[-1]
    
    # Handle array case
    if isinstance(prev_close, np.ndarray):
        prev_close = prev_close[-1] if len(prev_close) > 0 else prev_close.item()
        prev_high = prev_high[-1] if len(prev_high) > 0 else prev_high.item()
        prev_low = prev_low[-1] if len(prev_low) > 0 else prev_low.item()
    
    # Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # R1, S1, PP (Pivot Point)
    R1 = prev_close + range_val * 1.1 / 12
    S1 = prev_close - range_val * 1.1 / 12
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w EMA34 warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1w uptrend + volume spike
            long_signal = (close[i] > R1) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 1w downtrend + volume spike
            short_signal = (close[i] < S1) and trend_1w_downtrend and volume_spike[i]
            
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
            # Exit: price reverts to pivot point (PP) OR 1w trend turns down
            if (close[i] <= PP or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reverts to pivot point (PP) OR 1w trend turns up
            if (close[i] >= PP or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
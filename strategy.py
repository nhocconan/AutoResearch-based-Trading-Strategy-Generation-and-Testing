#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirm
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakout from prior 4h bar, combined with 4h EMA50 trend filter and volume confirmation.
R1/S1 are the most reliable Camarilla levels for intraday breakouts. 4h EMA50 ensures alignment with higher timeframe trend.
Volume spike confirms breakout strength. Session filter (08-20 UTC) reduces noise trades.
Designed for 15-30 trades/year (60-120 over 4 years) to avoid fee drag on 1h timeframe.
Works in bull markets via breakout continuation and bear markets via fade at extreme levels (confluence filters reduce false signals).
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter and Camarilla calculation (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Prior 4h bar OHLC for Camarilla calculation
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla levels: R1, S1 (most reliable for breakouts)
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R1/S1 breakout + volume spike + 4h EMA50 trend alignment
            long_breakout = curr_high > r1_aligned[i]
            short_breakout = curr_low < s1_aligned[i]
            
            # Trend filter: price must be on correct side of 4h EMA50
            long_trend = curr_close > ema_50_4h_aligned[i]
            short_trend = curr_close < ema_50_4h_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla R1 (failed breakout) or trend reverses
            if curr_close < r1_aligned[i] or curr_close < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price closes above Camarilla S1 (failed breakout) or trend reverses
            if curr_close > s1_aligned[i] or curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0
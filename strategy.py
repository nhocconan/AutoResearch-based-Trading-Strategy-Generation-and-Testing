#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_Filter
Hypothesis: 6-hour Camarilla R3/S3 breakout with 1-day EMA34 trend filter and volume confirmation.
Targets 12-37 trades/year by requiring: 1) price breaks daily R3/S3 levels (stronger breakout), 
2) aligned with 1d EMA34 trend, 3) volume > 1.8x 24-period average. Uses 6h timeframe to reduce 
overtrading while capturing significant moves. Volume filter reduces false breakouts. Designed to 
work in both bull and bear markets by following the 1d trend direction, avoiding counter-trend 
entries that fail in ranging/volatile conditions.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R3 and S3 levels (R3 = C + 1.1*(HL/2), S3 = C - 1.1*(HL/2))
    R3 = prev_close + 1.1 * prev_range * (1.0/2.0)
    S3 = prev_close - 1.1 * prev_range * (1.0/2.0)
    
    # Align 1d levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8 * 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA34 (34) + volume MA (24)
    start_idx = 34 + 24 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long breakout: price breaks above R3 with uptrend and volume confirmation
            long_breakout = (curr_close > R3_aligned[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below S3 with downtrend and volume confirmation
            short_breakout = (curr_close < S3_aligned[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below S3 (mean reversion) or trend changes to downtrend
            if curr_close < S3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R3 (mean reversion) or trend changes to uptrend
            if curr_close > R3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0
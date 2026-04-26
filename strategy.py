#!/usr/bin/env python3
"""
1h_HighLow_Breakout_4hTrend_VolumeFilter_Session
Hypothesis: Break above prior session high/low (08-20 UTC) with 4h EMA50 trend filter and volume spike (>1.8x 20-period average). Uses session filter to trade only during active UTC hours (08-20) to reduce noise. Targets 15-30 trades/year by requiring multi-bar confirmation and strict alignment. Works in bull/bear via trend filter and session-based structure.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Calculate prior session high/low (08-20 UTC)
    session_high = np.full(n, np.nan)
    session_low = np.full(n, np.nan)
    
    for i in range(n):
        if not in_session[i]:
            session_high[i] = session_high[i-1] if i > 0 else np.nan
            session_low[i] = session_low[i-1] if i > 0 else np.nan
            continue
            
        if i == 0 or not in_session[i-1]:
            # First bar of session
            session_high[i] = high[i]
            session_low[i] = low[i]
        else:
            # Continuing session
            session_high[i] = max(session_high[i-1], high[i])
            session_low[i] = min(session_low[i-1], low[i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA50 (50) + volume MA (20) + session setup
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(session_high[i]) or np.isnan(session_low[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: break above session high + 4h uptrend + volume spike
            # Require 2-bar confirmation to avoid false breakouts
            long_breakout = (close[i] > session_high[i]) and \
                           (close[i-1] > session_high[i-1])
            long_signal = long_breakout and trend_4h_uptrend and volume_spike[i]
            
            # Short: break below session low + 4h downtrend + volume spike
            short_breakout = (close[i] < session_low[i]) and \
                           (close[i-1] < session_low[i-1])
            short_signal = short_breakout and trend_4h_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below session low OR 4h trend turns down
            if (close[i] < session_low[i] or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above session high OR 4h trend turns up
            if (close[i] > session_high[i] or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HighLow_Breakout_4hTrend_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0
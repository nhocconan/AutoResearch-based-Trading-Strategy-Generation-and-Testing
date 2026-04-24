#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation, active only during 08-20 UTC session.
- Uses 1h timeframe (primary) and 4h HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 4h OHLC: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
- Breakout logic: long when price crosses above H3 with volume confirmation and uptrend, short when price crosses below L3 with volume confirmation and downtrend
- Trend filter: only long when price > 4h EMA50, only short when price < 4h EMA50
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volume false signals
- Session filter: trade only between 08:00-20:00 UTC to reduce noise and focus on liquid sessions
- Discrete signal size: 0.20 to minimize fee churn and manage drawdown
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior 4h Camarilla levels (H3 and L3)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 4h EMA50 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 4h close (mean reversion) or reverse signal
            prev_close_4h = df_4h['close'].shift(1).values
            prev_close_aligned = align_htf_to_ltf(prices, df_4h, prev_close_4h)
            if close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to prior 4h close (mean reversion) or reverse signal
            prev_close_4h = df_4h['close'].shift(1).values
            prev_close_aligned = align_htf_to_ltf(prices, df_4h, prev_close_4h)
            if close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0
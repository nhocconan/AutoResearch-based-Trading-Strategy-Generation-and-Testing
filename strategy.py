#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1w HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d session: H4 = 1.1/2*(high-low) + close, L4 = close - 1.1/2*(high-low)
- Breakout logic: long when price crosses above H4 with volume confirmation, short when price crosses below L4
- Trend filter: only long when price > 1w EMA50, only short when price < 1w EMA50
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 1d close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate prior 1d Camarilla H4/L4 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC to avoid look-ahead (today's Camarilla based on yesterday's range)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    cam_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    cam_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe (already delayed by shift(1))
    cam_h4_aligned = align_htf_to_ltf(prices, df_1d, cam_h4)
    cam_l4_aligned = align_htf_to_ltf(prices, df_1d, cam_l4)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    # Mean reversion exit: price reverts to prior 1d close
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1w EMA50 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(cam_h4_aligned[i]) or 
            np.isnan(cam_l4_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(prev_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H4 AND uptrend AND volume confirmation
            if close[i] > cam_h4_aligned[i] and close[i-1] <= cam_h4_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L4 AND downtrend AND volume confirmation
            elif close[i] < cam_l4_aligned[i] and close[i-1] >= cam_l4_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d close (mean reversion) or reverse signal
            if close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 1d close (mean reversion) or reverse signal
            if close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d session: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
- Breakout logic: long when price crosses above H3 with volume confirmation, short when price crosses below L3
- Trend filter: only long when price > 4h EMA50, only short when price < 4h EMA50
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 1d close (mean reversion)
- Discrete signal size: 0.20 to balance return and risk
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate prior 1d Camarilla H3/L3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC to avoid look-ahead (today's Camarilla based on yesterday's range)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    cam_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    cam_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 1h timeframe (already delayed by shift(1))
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    # Mean reversion exit: price reverts to prior 1d close
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 4h EMA50 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(cam_h3_aligned[i]) or 
            np.isnan(cam_l3_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(prev_close_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 AND uptrend AND volume confirmation
            if close[i] > cam_h3_aligned[i] and close[i-1] <= cam_h3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below L3 AND downtrend AND volume confirmation
            elif close[i] < cam_l3_aligned[i] and close[i-1] >= cam_l3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d close (mean reversion) or reverse signal
            if close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to prior 1d close (mean reversion) or reverse signal
            if close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0
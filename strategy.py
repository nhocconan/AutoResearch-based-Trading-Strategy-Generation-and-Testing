#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume spike confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA34 trend alignment
- Camarilla pivot levels (H3, L3, H4, L4) calculated from previous completed 4h bar
- Long when price breaks above H3 AND price > 4h EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below L3 AND price < 4h EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the 4h close (mean reversion to equilibrium)
- Discrete signal size: 0.20 to minimize fee churn
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla levels act as dynamic support/resistance
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA34 for trend filter (using previous completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla pivot levels (H3, L3, H4, L4) from previous completed 4h bar
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    #           L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    rng = high_4h - low_4h
    h4 = close_4h + 1.5 * rng
    h3 = close_4h + 1.125 * rng
    l3 = close_4h - 1.125 * rng
    l4 = close_4h - 1.5 * rng
    
    # Align Camarilla levels to 1h timeframe (previous completed 4h bar values)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 4h EMA34
    uptrend = close > ema_34_4h_aligned
    downtrend = close < ema_34_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 4h EMA34, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > h3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < l3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to 4h close (equilibrium)
            if close[i] < close_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to 4h close (equilibrium)
            if close[i] > close_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0
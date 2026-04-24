#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume spike (>1.8x 48-bar average).
- Targets Camarilla breakout levels (H4/L4) which are stronger than H3/L3 for continuation.
- Uses 12h EMA50 for trend filter to ensure alignment with higher timeframe direction.
- Volume confirmation ensures breakouts have conviction.
- Discrete position size 0.25 to manage drawdown and reduce fee churn.
- Designed for 6h timeframe to balance trade frequency and signal quality.
- Works in bull/bear markets: trend filter prevents counter-trend entries, volume filter avoids low-conviction breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h OHLC (completed 12h bar)
    high_12h = df_12h['high'].shift(1).values
    low_12h = df_12h['low'].shift(1).values
    close_12h = df_12h['close'].shift(1).values
    
    # Align to 6h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate Camarilla levels (H4/L4 for breakout continuation)
    camarilla_h4 = close_12h_aligned + 1.1 * (high_12h_aligned - low_12h_aligned) / 2
    camarilla_l4 = close_12h_aligned - 1.1 * (high_12h_aligned - low_12h_aligned) / 2
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 48-period average (more stringent than 24-bar)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 48)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H4 AND price above 12h EMA50 AND volume confirmation
            if close[i] > camarilla_h4[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 12h EMA50 AND volume confirmation
            elif close[i] < camarilla_l4[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L4 OR price crosses below 12h EMA50
            if close[i] < camarilla_l4[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H4 OR price crosses above 12h EMA50
            if close[i] > camarilla_h4[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
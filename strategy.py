#!/usr/bin/env python3
"""
1h_SuperTrend_Filtered_4hTrend_With_Volume_Confirmation
Hypothesis: Use 1h SuperTrend for entry timing, filtered by 4h trend direction (EMA50) and volume spikes.
Only trade in direction of 4h trend to avoid counter-trend whipsaws. Volume confirmation ensures momentum.
Designed for 15-30 trades/year to minimize fee drag. Works in bull via trend continuation and bear via trend reversals.
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
    
    # Calculate SuperTrend on 1h data
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + atr_multiplier * atr
    basic_lb = (high + low) / 2 - atr_multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    supertrend = np.zeros_like(close)
    supertrend[0] = final_ub[0]
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if close[i] > final_ub[i-1]:
            direction[i] = 1
        elif close[i] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 1.5 * 24-period average (on 1h data, ~1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for ATR, EMA, and volume average
    start_idx = max(atr_period, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(supertrend[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_confirm[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        st_value = supertrend[i]
        ema_4h_val = ema_50_4h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: SuperTrend uptrend (close > SuperTrend) AND price above 4h EMA50 AND volume confirmation
            if close[i] > st_value and close[i] > ema_4h_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: SuperTrend downtrend (close < SuperTrend) AND price below 4h EMA50 AND volume confirmation
            elif close[i] < st_value and close[i] < ema_4h_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: SuperTrend flips to downtrend (close < SuperTrend)
            if close[i] < st_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: SuperTrend flips to uptrend (close > SuperTrend)
            if close[i] > st_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_SuperTrend_Filtered_4hTrend_With_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0
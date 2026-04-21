#!/usr/bin/env python3
"""
4h_MidlineBreakout_Pullback
Hypothesis: Buy pullbacks to the midline (median of recent 20-period high-low range) in the direction of the 1-week EMA50 trend, confirmed by volume > 1.5x average. This strategy captures mean-reversion within a trend, which works in both bull (buy dips) and bear (sell rallies) markets. The 1w EMA50 ensures we trade with the dominant weekly trend, reducing counter-trend trades. Volume confirmation ensures the move has participation. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros_like(close_1w)
    ema50_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align 1w EMA50 to 4h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-period ATR for volatility (used in midline calculation)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 4:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-4:i])
    
    # 20-period rolling max and min for midline calculation
    roll_max = np.zeros_like(high)
    roll_min = np.zeros_like(low)
    for i in range(n):
        if i < 20:
            roll_max[i] = np.max(high[:i+1])
            roll_min[i] = np.min(low[:i+1])
        else:
            roll_max[i] = np.max(high[i-20+1:i+1])
            roll_min[i] = np.min(low[i-20+1:i+1])
    
    # Midline = average of 20-period high and low
    midline = (roll_max + roll_min) / 2.0
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1])
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(midline[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        mid = midline[i]
        vol_ok = volume_filter[i]
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to or below midline in uptrend (price > 1w EMA50) with volume
            if price <= mid and price > ema50 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price rallies to or above midline in downtrend (price < 1w EMA50) with volume
            elif price >= mid and price < ema50 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses above midline (mean reversion complete) or trend breaks
            if price > mid or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below midline (mean reversion complete) or trend breaks
            if price < mid or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MidlineBreakout_Pullback"
timeframe = "4h"
leverage = 1.0
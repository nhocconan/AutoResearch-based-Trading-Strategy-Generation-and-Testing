#!/usr/bin/env python3
"""
1d_1w_WeeklyTrend_DailyBreakout
Hypothesis: On 1d timeframe, trade breakouts of 20-day high/low only when aligned with 1-week trend (price above/below 20-week EMA) and with volume confirmation (>1.5x average). This captures strong momentum moves in both bull and bear markets while avoiding counter-trend trades. Low trade frequency expected (target: 10-25/year) due to dual timeframe alignment requirement. Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros_like(close_1w)
    ema20_1w[0] = close_1w[0]
    alpha = 2.0 / (20 + 1)
    for i in range(1, len(close_1w)):
        ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Align 20-week EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day high/low for breakout levels
    highest_20d = np.full(n, np.nan)
    lowest_20d = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        highest_20d[i] = np.max(high[start_idx:i+1])
        lowest_20d[i] = np.min(low[start_idx:i+1])
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        volume_avg[i] = np.mean(volume[start_idx:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR for stoploss (10-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 9)
        atr[i] = np.mean(tr[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after 20-day warmup
        # Skip if NaN in critical values
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(highest_20d[i]) or np.isnan(lowest_20d[i]) or np.isnan(volume_filter[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20w = ema20_1w_aligned[i]
        high_20d = highest_20d[i]
        low_20d = lowest_20d[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: break above 20-day high with volume and weekly uptrend (price > 20w EMA)
            if price > high_20d and vol_ok and price > ema20w:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below 20-day low with volume and weekly downtrend (price < 20w EMA)
            elif price < low_20d and vol_ok and price < ema20w:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below 20-day low or breaks below 20-week EMA
            if price < low_20d or price < ema20w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above 20-day high or breaks above 20-week EMA
            if price > high_20d or price > ema20w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_WeeklyTrend_DailyBreakout"
timeframe = "1d"
leverage = 1.0
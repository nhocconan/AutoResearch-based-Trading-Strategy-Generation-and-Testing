#!/usr/bin/env python3
"""
1d_1w_Keltner_Breakout_Trend
Hypothesis: On 1d timeframe, break above/below Keltner Channel (20, 2.0) signals trend.
Use 1-week EMA50 as trend filter: only take longs when price > 1w EMA50, shorts when price < 1w EMA50.
This avoids counter-trend trades in strong monthly trends. Weekly EMA50 adapts slowly,
providing a robust bull/bear filter. Keltner breakout catches momentum; weekly filter
ensures alignment with major trend. Target: 15-25 trades/year (60-100 total).
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
"""

name = "1d_1w_Keltner_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 1d Keltner Channel (20, 2.0) ---
    # EMA20 of close
    ema_20 = np.full(n, np.nan)
    if n >= 20:
        ema_20[19] = np.mean(close[:20])
        for i in range(20, n):
            ema_20[i] = 0.1 * close[i] + 0.9 * ema_20[i-1]
    
    # ATR(20)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first TR
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[1:21])  # average of TR[1] to TR[20]
        for i in range(21, n):
            atr[i] = 0.05 * tr[i] + 0.95 * atr[i-1]
    
    # Keltner bounds
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # --- 1w EMA50 (trend filter) ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = 2/(50+1) * close_1w[i] + (1 - 2/(50+1)) * ema_1w[i-1]
    
    # Align 1w EMA50 to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d EMA20, ATR20, 1w EMA50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > kc_upper[i]
        breakout_down = close[i] < kc_lower[i]
        
        # 1w trend: price above/below EMA50
        trend_up = close[i] > ema_1w_aligned[i]  # Compare 1d close to 1w EMA50 aligned
        trend_down = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            if breakout_up and trend_up:
                # Long: breakout above KC upper in 1w uptrend
                signals[i] = 0.25
                position = 1
            elif breakout_down and trend_down:
                # Short: breakdown below KC lower in 1w downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: break below KC lower OR trend turns down
                if breakout_down or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: break above KC upper OR trend turns up
                if breakout_up or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
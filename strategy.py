#!/usr/bin/env python3
"""
1h_PriceAction_HighLow_Breakout_4hTrend_Direction
Hypothesis: Price breaks above recent 10-period high or below 10-period low on 1h,
only in direction of 4h trend (EMA50), with volume confirmation. Uses 4h trend for
direction to avoid whipsaws, 1h for entry timing. Session filter (08-20 UTC) to reduce noise.
Target: 15-35 trades/year per symbol. Works in bull/bear by following 4h trend.
"""

name = "1h_PriceAction_HighLow_Breakout_4hTrend_Direction"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h highest high and lowest low of last 10 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(10, n):
        highest_high[i] = np.max(high[i-10:i])
        lowest_low[i] = np.min(low[i-10:i])
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10)  # EMA + volume + HH/LL warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above 10-period high AND above 4h EMA50
            if close[i] > highest_high[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Break below 10-period low AND below 4h EMA50
            elif close[i] < lowest_low[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Close below 4h EMA50
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Close above 4h EMA50
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
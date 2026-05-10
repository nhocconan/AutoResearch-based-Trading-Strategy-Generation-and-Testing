#!/usr/bin/env python3
# 1d_VWAP_Reversion_with_WeeklyTrend
# Hypothesis: Mean reversion to daily VWAP with weekly trend filter captures mean-reversion
# in ranging markets while avoiding trades against the weekly trend. VWAP provides a dynamic
# mean-reversion level, and weekly EMA50 filters direction to reduce false signals.
# Designed for low trade frequency (10-20/year) to minimize fee drift.

name = "1d_VWAP_Reversion_with_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Distance from VWAP as percentage
    dist_from_vwap = (close - vwap) / vwap * 100.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need weekly EMA50 warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vwap[i]) or vwap[i] == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: above weekly EMA = uptrend, below = downtrend
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long when price is significantly below VWAP in uptrend (mean reversion up)
            if dist_from_vwap[i] <= -1.5 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short when price is significantly above VWAP in downtrend (mean reversion down)
            elif dist_from_vwap[i] >= 1.5 and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price returns to VWAP or breaks above +1.0 (take profit)
            if dist_from_vwap[i] >= -0.5 or dist_from_vwap[i] >= 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price returns to VWAP or breaks below -1.0 (take profit)
            if dist_from_vwap[i] <= 0.5 or dist_from_vwap[i] <= -1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
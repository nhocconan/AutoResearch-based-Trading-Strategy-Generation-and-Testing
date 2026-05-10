#!/usr/bin/env python3
# 6h_LongTermTrend_Pullback_With_1dTrend
# Hypothesis: In multi-year crypto markets, strong trends persist across timeframes.
# We use 1-day EMA50 as the primary trend filter and enter on 6-hour pullbacks to EMA20.
# Long when: 1d trend up (close > EMA50_1d) AND price pulls back to touch EMA20_6h from below.
# Short when: 1d trend down (close < EMA50_1d) AND price pulls back to touch EMA20_6h from above.
# This captures continuation moves in trending markets while avoiding counter-trend trades.
# Works in both bull (follows strong uptrends) and bear (follows strong downtrends).
# Uses volume confirmation to avoid low-conviction breakouts.

name = "6h_LongTermTrend_Pullback_With_1dTrend"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA20 on 6h chart for pullback entries
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (24-period MA on 6h chart = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), EMA50_1d (50), volume MA (24)
    start_idx = max(20, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to EMA20 for pullback detection
        # For long: price crosses above EMA20 from below (pullback in uptrend)
        # For short: price crosses below EMA20 from above (pullback in downtrend)
        if i > 0:
            cross_above_ema20 = (close[i] > ema_20[i]) and (close[i-1] <= ema_20[i-1])
            cross_below_ema20 = (close[i] < ema_20[i]) and (close[i-1] >= ema_20[i-1])
        else:
            cross_above_ema20 = False
            cross_below_ema20 = False
        
        if position == 0:
            # Long entry: uptrend + pullback to EMA20 from below + volume
            if uptrend and cross_above_ema20 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + pullback to EMA20 from above + volume
            elif downtrend and cross_below_ema20 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or reversal signal
            if not uptrend or cross_below_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or reversal signal
            if not downtrend or cross_above_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
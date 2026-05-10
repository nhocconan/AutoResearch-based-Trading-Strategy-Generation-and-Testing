#!/usr/bin/env python3
# 1d_WeeklyTrend_Pullback_With_VolumeConfirmation
# Hypothesis: In multi-year crypto markets, strong weekly trends persist and provide reliable entry signals.
# We use weekly EMA20 as the primary trend filter and enter on daily pullbacks to EMA50.
# Long when: weekly trend up (close > EMA20_weekly) AND price pulls back to touch EMA50 from below.
# Short when: weekly trend down (close < EMA20_weekly) AND price pulls back to touch EMA50 from above.
# This captures continuation moves in trending markets while avoiding counter-trend trades.
# Works in both bull (follows strong uptrends) and bear (follows strong downtrends).
# Uses volume confirmation to avoid low-conviction breakouts.

name = "1d_WeeklyTrend_Pullback_With_VolumeConfirmation"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on daily chart for pullback entries
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation (20-period MA on daily chart = ~1 month)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), EMA20_1w (20), volume MA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to EMA50 for pullback detection
        # For long: price crosses above EMA50 from below (pullback in uptrend)
        # For short: price crosses below EMA50 from above (pullback in downtrend)
        if i > 0:
            cross_above_ema50 = (close[i] > ema_50[i]) and (close[i-1] <= ema_50[i-1])
            cross_below_ema50 = (close[i] < ema_50[i]) and (close[i-1] >= ema_50[i-1])
        else:
            cross_above_ema50 = False
            cross_below_ema50 = False
        
        if position == 0:
            # Long entry: uptrend + pullback to EMA50 from below + volume
            if uptrend and cross_above_ema50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + pullback to EMA50 from above + volume
            elif downtrend and cross_below_ema50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or reversal signal
            if not uptrend or cross_below_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or reversal signal
            if not downtrend or cross_above_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band breakout with 4h trend filter and session filter.
# Long when price closes above BB(20,2) upper band AND 4h EMA20 > EMA50 (uptrend) AND hour in [8,20) UTC.
# Short when price closes below BB(20,2) lower band AND 4h EMA20 < EMA50 (downtrend) AND hour in [8,20) UTC.
# Exit when price crosses back inside BB(20,2) bands.
# Uses Bollinger Bands for volatility breakouts with trend filter to avoid false signals in chop.
# Target: 60-150 total trades over 4 years (15-37/year) for low fee drift.

name = "1h_BB_4hTrend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: 1 for uptrend (EMA20 > EMA50), -1 for downtrend (EMA20 < EMA50)
    trend_4h = np.where(ema20_4h > ema50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours < 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(trend_4h_aligned[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close above BB upper, uptrend, in session
            long_cond = (close[i] > bb_upper[i]) and (trend_4h_aligned[i] == 1) and session_filter[i]
            # Short conditions: close below BB lower, downtrend, in session
            short_cond = (close[i] < bb_lower[i]) and (trend_4h_aligned[i] == -1) and session_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: close back inside BB (below upper band)
            if close[i] < bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: close back inside BB (above lower band)
            if close[i] > bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
# 1h_ema_cross_htf_trend_v1
# Hypothesis: 1h EMA(9/21) crossover filtered by 4h EMA50 trend and 1d EMA200 trend filter.
# Only take longs when both HTF trends are bullish (price > EMA), shorts when both bearish.
# Uses 1h timeframe for entry timing but HTF for signal direction to reduce overtrading.
# Session filter (08-20 UTC) avoids low-volume Asian session noise.
# Target: 15-37 trades/year (60-150 total over 4 years) with discrete sizing 0.20.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_cross_htf_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d HTF data for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h EMA(9) and EMA(21) for entry timing
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session or missing data
        if not in_session[i] or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(ema9[i]) or np.isnan(ema21[i]):
            signals[i] = 0.0
            continue
        
        # Determine HTF trend: bullish if price > both EMAs, bearish if price < both
        htf_bullish = close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]
        htf_bearish = close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA crossover turns bearish OR HTF trend turns mixed/bearish
            if ema9[i] < ema21[i] or not htf_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA crossover turns bullish OR HTF trend turns mixed/bullish
            if ema9[i] > ema21[i] or not htf_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: bullish EMA crossover + bullish HTF trend
            if ema9[i] > ema21[i] and htf_bullish:
                position = 1
                signals[i] = 0.20
            # Enter short: bearish EMA crossover + bearish HTF trend
            elif ema9[i] < ema21[i] and htf_bearish:
                position = -1
                signals[i] = -0.20
    
    return signals
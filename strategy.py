#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Breakout above Camarilla R1 or below S1 on daily timeframe,
# filtered by weekly trend (price above/below weekly EMA50) and volume spike.
# Uses weekly EMA50 for trend filter and daily volume spike for confirmation.
# Designed to work in both bull and bear markets by following the weekly trend.
# Targets ~10-20 trades/year to minimize fee drag.

name = "1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter"
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
    
    # 1d data for Camarilla levels
    df_1d = prices  # already daily
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for today using yesterday's OHLC
        if i == 0:
            continue
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        range_prev = high_prev - low_prev
        
        # Camarilla levels
        R1 = close_prev + range_prev * 1.1 / 12
        S1 = close_prev - range_prev * 1.1 / 12
        
        if position == 0:
            # Long: breakout above R1 with weekly uptrend and volume spike
            if (close[i] > R1 and
                trend_1w_up_aligned[i] > 0.5 and
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 with weekly downtrend and volume spike
            elif (close[i] < S1 and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below R1 or weekly trend turns down
            if (close[i] < R1 or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above S1 or weekly trend turns up
            if (close[i] > S1 or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
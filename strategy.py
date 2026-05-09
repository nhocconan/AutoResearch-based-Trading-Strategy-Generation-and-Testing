#!/usr/bin/env python3
# 1D_Weekly_Trend_With_Daily_Pullback
# Hypothesis: Uses weekly trend filter (price above/below weekly SMA50) and enters on daily pullbacks
# to the 21 EMA in the direction of the weekly trend. Works in bull markets (buy dips) and bear markets
# (sell rallies) by following the higher timeframe trend. Weekly trend reduces whipsaw, daily EMA
# provides precise entry. Target: 15-25 trades/year per symbol.

name = "1D_Weekly_Trend_With_Daily_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly SMA50 for trend filter
    sma_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        sma_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            sma_50_1w[i] = (sma_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    # Align weekly trend to daily
    weekly_trend = align_htf_to_ltf(prices, df_1w, close_1w > sma_50_1w)
    
    # Daily EMA21 for pullback entries
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly SMA50
    
    for i in range(start_idx, n):
        # Skip if weekly trend not ready
        if np.isnan(weekly_trend[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: True = uptrend (price > SMA50), False = downtrend
        is_uptrend = weekly_trend[i]
        
        if position == 0:
            # Enter long: weekly uptrend AND price pulls back to EMA21
            if is_uptrend and close[i] <= ema_21[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend AND price rallies to EMA21
            elif not is_uptrend and close[i] >= ema_21[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down OR price moves above EMA21 (pullback complete)
            if not is_uptrend or close[i] > ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up OR price moves below EMA21 (pullback complete)
            if is_uptrend or close[i] < ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
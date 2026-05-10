#!/usr/bin/env python3
# 1h_4h_1d_Trend_Filtered_Momentum
# Hypothesis: Combines 4h trend (EMA21 vs EMA50) and 1d momentum (ROC10) to filter 1h breakouts.
# Enters long when 1h price breaks above 4h EMA21 with 4h uptrend and positive 1d momentum.
# Enters short when 1h price breaks below 4h EMA21 with 4h downtrend and negative 1d momentum.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours. Position size 0.20.
# Designed for 15-30 trades/year to minimize fee drag while capturing trending moves.

name = "1h_4h_1d_Trend_Filtered_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA21 and EMA50 for trend
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = ema21_4h > ema50_4h
    trend_4h_down = ema21_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d ROC(10) for momentum
    close_1d = df_1d['close'].values
    roc_1d = np.zeros_like(close_1d)
    roc_1d[10:] = (close_1d[10:] - close_1d[:-10]) / close_1d[:-10] * 100
    
    # Align 1d ROC to 1h
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # warmup period
        # Skip if data not ready
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(roc_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 4h EMA21, 4h uptrend, positive 1d momentum
            if (close[i] > ema21_4h_aligned[i] and 
                trend_4h_up_aligned[i] > 0.5 and 
                roc_1d_aligned[i] > 0):
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA21, 4h downtrend, negative 1d momentum
            elif (close[i] < ema21_4h_aligned[i] and 
                  trend_4h_down_aligned[i] > 0.5 and 
                  roc_1d_aligned[i] < 0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when price crosses below 4h EMA21 or trend fails
            if (close[i] < ema21_4h_aligned[i] or 
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when price crosses above 4h EMA21 or trend fails
            if (close[i] > ema21_4h_aligned[i] or 
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
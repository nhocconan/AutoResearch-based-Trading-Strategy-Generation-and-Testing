#!/usr/bin/env python3
# 1h_4h_1d_ema_cross_volume_v1
# Strategy: 1h EMA cross with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: 1h EMA cross (9/21) provides timely entries, filtered by 4h EMA50 trend and 1d EMA200 trend.
# Volume spike (1.5x 20-period average) confirms momentum. Designed for 15-30 trades/year to avoid fee drag.
# Works in bull/bear by requiring alignment with higher timeframe trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_ema_cross_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h EMA(9) and EMA(21) for entry signal
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema9.iloc[i]) or np.isnan(ema21.iloc[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters: 4h and 1d trends
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_200_1d_aligned[i]
        downtrend_1d = close[i] < ema_200_1d_aligned[i]
        
        # EMA cross signals
        ema9_now = ema9.iloc[i]
        ema21_now = ema21.iloc[i]
        ema9_prev = ema9.iloc[i-1]
        ema21_prev = ema21.iloc[i-1]
        ema_cross_bull = ema9_now > ema21_now and ema9_prev <= ema21_prev
        ema_cross_bear = ema9_now < ema21_now and ema9_prev >= ema21_prev
        
        # Entry logic: EMA cross + volume spike + trend alignment (both 4h and 1d must agree)
        if (ema_cross_bull and volume_spike[i] and uptrend_4h and uptrend_1d and position != 1):
            position = 1
            signals[i] = 0.20
        elif (ema_cross_bear and volume_spike[i] and downtrend_4h and downtrend_1d and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: EMA cross reversal or trend disagreement
        elif position == 1 and (ema_cross_bear or not (uptrend_4h and uptrend_1d)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema_cross_bull or not (downtrend_4h and downtrend_1d)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals
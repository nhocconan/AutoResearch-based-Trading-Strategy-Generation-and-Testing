#!/usr/bin/env python3
# 12h_1d_w_trend_following_v1
# Strategy: 12h timeframe with 1d weekly trend filter and ATR-based volatility filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: In trending markets (price above/below 1d EMA50), follow the trend with entries on pullbacks to the 12h EMA21. In ranging markets (price near 1d EMA50), avoid trades. Uses ATR to filter low-volatility chop. Designed for low frequency (~20-40/year) to minimize fee drag while capturing major trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_w_trend_following_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = np.inf
    tr2[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h EMA(21) for trend following
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR-based volatility filter: current ATR > 0.5 * 1d ATR
    atr_12 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_12[0] = np.nan
    vol_filter = atr_12 > (0.5 * atr_14_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: pullback to 12h EMA21 in trending market with volatility filter
        if uptrend and close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99 and vol_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif downtrend and close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99 and vol_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or low volatility
        elif position == 1 and (not uptrend or not vol_filter[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or not vol_filter[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
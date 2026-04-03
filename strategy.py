#!/usr/bin/env python3
"""
Experiment #1874: 1h Donchian Breakout + Volume + HTF Trend Filter
HYPOTHESIS: 1h Donchian(20) breakouts with volume confirmation (>1.5x average) and 4h/1d trend alignment capture sustained moves while avoiding false breakouts. Uses 4h EMA(50) and 1d EMA(50) for trend filter - only long when both HTF EMAs are above price, short when both below. Discrete position sizing of 0.20 to manage drawdown and reduce fee churn. Session filter (08-20 UTC) avoids low-volume Asian session. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1874_1h_donchian_vol_htf_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for EMA(50) trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for EMA(50) trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any indicator is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: require both 4h and 1d to agree
        # Long bias: both HTF trends bullish (> 0)
        # Short bias: both HTF trends bearish (< 0)
        long_bias = (trend_4h_aligned[i] > 0) and (trend_1d_aligned[i] > 0)
        short_bias = (trend_4h_aligned[i] < 0) and (trend_1d_aligned[i] < 0)
        
        # Volume confirmation: require > 1.5x average volume
        volume_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > highest[i-1]  # break above previous period high
        breakout_short = price < lowest[i-1]  # break below previous period low
        
        # Entry logic
        if long_bias and volume_confirm and breakout_long:
            signals[i] = SIZE
        elif short_bias and volume_confirm and breakout_short:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
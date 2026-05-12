#!/usr/bin/env python3
name = "1d_TRIX_WeeklyTrend_VolumeFilter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (15-period triple EMA rate of change)
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1))
    trix = trix.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False).mean().values
    
    # Weekly trend filter (from 1w)
    df_1w = get_htf_data(prices, '1w')
    close_w = df_1w['close'].values
    ema13_w = pd.Series(close_w).ewm(span=13, adjust=False).mean().values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False).mean().values
    weekly_trend_up = ema13_w > ema34_w
    weekly_trend_down = ema13_w < ema34_w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Volume filter: current volume > 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]) or np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal + weekly uptrend + volume filter
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and weekly_trend_up_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal + weekly downtrend + volume filter
            elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and weekly_trend_down_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below signal OR weekly trend changes
            if trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above signal OR weekly trend changes
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
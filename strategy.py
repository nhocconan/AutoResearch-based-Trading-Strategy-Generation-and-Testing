#!/usr/bin/env python3
# 1d_TRIX_VolumeSpike_1wTrend
# Hypothesis: TRIX momentum with volume spike confirmation and weekly trend filter for 1d timeframe.
# TRIX (12) captures momentum shifts, volume spike confirms breakout strength, weekly EMA (21) filters trend direction.
# Designed to work in both bull and bear markets by following the weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) with low frequency to minimize fee drag.

name = "1d_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # TRIX (12) calculation: triple EMA of log returns
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    ema1 = pd.Series(log_returns).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (np.diff(ema3, prepend=ema3[0]) / ema3)
    
    # Weekly volume spike: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1w = mean_arr(volume_1w, 20)
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX turns positive, above weekly EMA21, strong volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_21_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns negative, below weekly EMA21, strong volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_21_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or price drops below weekly EMA21
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or price rises above weekly EMA21
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
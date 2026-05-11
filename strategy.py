#!/usr/bin/env python3
name = "1d_TrixVolumeSpike_1wTrend"
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
    
    # Get 1w data for trend filter (weekly EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema20_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # TRIX calculation on daily: EMA of EMA of EMA of log returns
    # TRIX = EMA(EMA(EMA(log(close), period), period), period)
    log_close = np.log(close)
    ema1 = pd.Series(log_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)  # percentage change
    trix[0] = 0  # first value undefined
    
    # Volume spike: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix[i]) or np.isnan(trend_up_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX positive + weekly uptrend + volume spike
            if trix[i] > 0 and trend_up_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative + weekly downtrend + volume spike
            elif trix[i] < 0 and not trend_up_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative OR weekly trend turns down
            if trix[i] < 0 or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive OR weekly trend turns up
            if trix[i] > 0 or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
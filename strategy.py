#!/usr/bin/env python3
name = "12h_TRIX_VolumeSpike_1dTrend"
timeframe = "12h"
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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # TRIX (12,9): TRIX = EMA(EMA(EMA(close,12),12),12); signal = 9-period EMA of TRIX
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1))
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal
    trix_hist = trix_hist.fillna(0).values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for TRIX and EMAs
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(trix_hist[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX histogram crosses above zero + daily uptrend + volume filter
            if trix_hist[i] > 0 and trix_hist[i-1] <= 0 and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX histogram crosses below zero + daily downtrend + volume filter
            elif trix_hist[i] < 0 and trix_hist[i-1] >= 0 and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX histogram crosses below zero or daily trend down
            if trix_hist[i] < 0 and trix_hist[i-1] >= 0 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX histogram crosses above zero or daily trend up
            if trix_hist[i] > 0 and trix_hist[i-1] <= 0 or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
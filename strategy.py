#!/usr/bin/env python3
name = "1d_1w_TRIX_Trend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w TRIX (15-period) for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (ema3_today - ema3_yesterday) / ema3_yesterday
    trix_raw = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix = np.concatenate([[np.nan], trix_raw])
    trix_1w = align_htf_to_ltf(prices, df_1w, trix)
    
    # Daily volume filter: volume > 1.3x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.3 * vol_ma20_1d_aligned
    
    # Daily EMA50 for additional trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for TRIX, EMA50, and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix_1w[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(ema50[i])):
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
            # Long: TRIX positive (rising momentum) + price above EMA50 + volume filter
            if trix_1w[i] > 0 and close[i] > ema50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative (falling momentum) + price below EMA50 + volume filter
            elif trix_1w[i] < 0 and close[i] < ema50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or price below EMA50
            if trix_1w[i] < 0 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or price above EMA50
            if trix_1w[i] > 0 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
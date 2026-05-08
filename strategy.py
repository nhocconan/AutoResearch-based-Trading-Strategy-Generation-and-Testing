#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR on 1d data for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0.0], tr])  # align length
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct_1d = atr_1d / close_1d
    atr_pct_ma_1d = pd.Series(atr_pct_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = atr_pct_1d / atr_pct_ma_1d
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    
    # Align HTF indicators to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Pre-compute session filter (8:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h EMA21 > EMA50 + volatility above average
            if (ema_21_4h_aligned[i] > ema_50_4h_aligned[i] and vol_ratio_1d_aligned[i] > 1.2):
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA21 < EMA50 + volatility above average
            elif (ema_21_4h_aligned[i] < ema_50_4h_aligned[i] and vol_ratio_1d_aligned[i] > 1.2):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h EMA21 crosses below EMA50
            if ema_21_4h_aligned[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h EMA21 crosses above EMA50
            if ema_21_4h_aligned[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
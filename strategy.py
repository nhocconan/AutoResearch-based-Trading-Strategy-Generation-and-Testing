#!/usr/bin/env python3
name = "1h_4h1d_HybridTrendBreakout_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR14 for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1h session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # 1h range breakout (previous 20-bar range)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need 20 for range, 14 for ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip outside session
        if not session_ok[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 20-bar high, above 4h EMA20, and volatility filter
            if close[i] > high_20[i] and close[i] > ema_20_4h_aligned[i] and volume[i] > 0:
                atr = atr_14_1d_aligned[i]
                if atr > 0 and volume[i] > np.median(volume[max(0, i-20):i+1]) * 1.5:
                    signals[i] = 0.20
                    position = 1
            # Short: break below 20-bar low, below 4h EMA20, and volatility filter
            elif close[i] < low_20[i] and close[i] < ema_20_4h_aligned[i] and volume[i] > 0:
                atr = atr_14_1d_aligned[i]
                if atr > 0 and volume[i] > np.median(volume[max(0, i-20):i+1]) * 1.5:
                    signals[i] = -0.20
                    position = -1
        elif position != 0:
            # Exit: price returns to 20-bar range or breaks in opposite direction
            if position == 1:
                if close[i] < low_20[i] or close[i] < ema_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > high_20[i] or close[i] > ema_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals
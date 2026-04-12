#/usr/bin/env python3
"""
1h_4d_cci_trend
Uses CCI(20) on 4h timeframe to identify trend direction.
Enters on 1h when CCI crosses above +100 (long) or below -100 (short) with volume confirmation.
Exits when CCI returns to zero line.
Uses 1d ADX(14) > 25 as trend strength filter to avoid ranging markets.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in trending markets by following CCI extremes.
"""

name = "1h_4d_cci_trend"
timeframe = "1h"
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
    
    # Get 4h data for CCI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate CCI(20) on 4h
    tp_4h = (high_4h + low_4h + close_4h) / 3.0  # typical price
    sma_tp = pd.Series(tp_4h).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_4h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    cci_4h = np.where(mad != 0, (tp_4h - sma_tp) / (0.015 * mad), 0.0)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 1h timeframe
    cci_4h_aligned = align_htf_to_ltf(prices, df_4h, cci_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 1h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(cci_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check filters
        trend_filter = adx_1d_aligned[i] > 25
        session_ok = session_filter[i]
        
        # Long entry: CCI crosses above +100 with volume and session
        if (cci_4h_aligned[i] > 100 and 
            (i == 100 or cci_4h_aligned[i-1] <= 100) and  # crossover
            vol_confirm[i] and session_ok and trend_filter and position != 1):
            position = 1
            signals[i] = 0.20
        # Short entry: CCI crosses below -100 with volume and session
        elif (cci_4h_aligned[i] < -100 and 
              (i == 100 or cci_4h_aligned[i-1] >= -100) and  # crossover
              vol_confirm[i] and session_ok and trend_filter and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit conditions: CCI returns to zero line
        elif position == 1 and cci_4h_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_4h_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals
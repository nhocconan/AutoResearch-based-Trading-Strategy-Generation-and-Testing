#!/usr/bin/env python3
"""
1h_4h_1d_CCI_MeanReversion
Strategy: Mean reversion at CCI extremes with trend filter.
- Uses 4h CCI(20) for mean reversion signals (long when CCI < -100, short when CCI > +100)
- 1d EMA34/100 filter to align with higher timeframe trend
- Volume confirmation (1.5x 20-bar average)
- Session filter: 08-20 UTC to avoid low liquidity periods
- Fixed position size: 0.20
Designed to work in both bull and bear markets by fading extremes in the direction of higher timeframe trend.
Timeframe: 1h
"""

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
    
    # Calculate 4h CCI(20)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price and moving average
    tp_4h = (high_4h + low_4h + close_4h) / 3.0
    tp_ma = pd.Series(tp_4h).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    tp_md = pd.Series(tp_4h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # CCI calculation
    cci_4h = (tp_4h - tp_ma) / (0.015 * tp_md)
    cci_4h = np.where(tp_md > 0, cci_4h, 0.0)  # Avoid division by zero
    
    # Align 4h CCI to 1h timeframe
    cci_4h_aligned = align_htf_to_ltf(prices, df_4h, cci_4h)
    
    # Calculate 1d EMA34 and EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema100_1d = close_series_1d.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d EMAs to 1h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume confirmation (1.5x 20-bar average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Apply filters
        vol_filter = volume[i] > (1.5 * volume_ma20[i])
        sess_filter = session_filter[i]
        
        if not (vol_filter and sess_filter):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filter from 1d EMAs
        uptrend = ema34_1d_aligned[i] > ema100_1d_aligned[i]
        downtrend = ema34_1d_aligned[i] < ema100_1d_aligned[i]
        
        # CCI mean reversion signals
        cci_oversold = cci_4h_aligned[i] < -100
        cci_overbought = cci_4h_aligned[i] > 100
        
        if position == 0:
            # Long: CCI oversold + uptrend
            if cci_oversold and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: CCI overbought + downtrend
            elif cci_overbought and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: CCI returns to neutral or trend changes
            if cci_4h_aligned[i] >= -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: CCI returns to neutral or trend changes
            if cci_4h_aligned[i] <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_CCI_MeanReversion"
timeframe = "1h"
leverage = 1.0
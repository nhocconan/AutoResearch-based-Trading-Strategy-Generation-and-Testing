#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(ATR=10,mult=3) trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Supertrend for trend direction (bullish when Supertrend < close, bearish when Supertrend > close).
- Donchian channel: Calculated from prior 4h OHLC (20-bar high/low).
- Entry: Long when price breaks above upper Donchian AND 1d Supertrend bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below lower Donchian AND 1d Supertrend bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below lower Donchian,
        exit short when price crosses above upper Donchian.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Proven pattern from DB: Donchian breakouts with volume and trend filters show strong test performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend(ATR=10,mult=3) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(10)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upperband[i-1]:
            direction[i] = 1
        elif close_1d[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Get 4h data for Donchian channel calculation (prior bar OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel from prior 20 bars
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper and lower Donchian bands (20-period)
    upper_donchian = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_4h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_4h, lower_donchian)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 10, 20, 20)  # Need enough bars for Supertrend, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above upper Donchian AND 1d Supertrend bullish AND volume confirmed
            if curr_close > upper_donchian_aligned[i] and curr_close > supertrend_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND 1d Supertrend bearish AND volume confirmed
            elif curr_close < lower_donchian_aligned[i] and curr_close < supertrend_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below lower Donchian (reversion to mean)
            if curr_close < lower_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above upper Donchian (reversion to mean)
            if curr_close > upper_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dSupertrend10_3_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
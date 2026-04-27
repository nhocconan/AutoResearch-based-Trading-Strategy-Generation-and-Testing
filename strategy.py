#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume filter and 1w trend filter.
Enters long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA)
with above-average daily volume and weekly uptrend.
Enters short when jaws cross below teeth with above-average daily volume and weekly downtrend.
Uses weekly timeframe for trend, daily for volume filter, 12h for execution.
Williams Alligator uses smoothed moving averages (SMMA) which reduce whipsaw in ranging markets.
Designed to work in both bull and bear markets by following the weekly trend and requiring volume confirmation.
Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 12h timeframe using SMMA
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator SMMA (13-period), daily volume MA, weekly EMA
    start_idx = max(13, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_34_1w_aligned[i]
        
        # Current Alligator values
        jaws_now = jaws[i]
        teeth_now = teeth[i]
        lips_now = lips[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Williams Alligator crossover with volume and weekly trend alignment
        if position == 0:
            # Long: jaws cross above teeth with volume + weekly uptrend
            if jaws_now > teeth_now and jaws[i-1] <= teeth[i-1] and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: jaws cross below teeth with volume + weekly downtrend
            elif jaws_now < teeth_now and jaws[i-1] >= teeth[i-1] and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: jaws cross below teeth or weekly trend turns down
            if jaws_now < teeth_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: jaws cross above teeth or weekly trend turns up
            if jaws_now > teeth_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0
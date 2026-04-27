#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with weekly trend filter and volume confirmation.
In oversold conditions (Williams %R < -80) with weekly uptrend: long.
In overbought conditions (Williams %R > -20) with weekly downtrend: short.
Williams %R identifies reversal points, weekly trend filters for direction,
volume confirms participation. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, length=14):
    """Williams %R: momentum oscillator measuring overbought/oversold levels"""
    if len(high) < length:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    highest_high = np.full_like(high, np.nan, dtype=np.float64)
    lowest_low = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(length-1, len(high)):
        highest_high[i] = np.max(high[i-length+1:i+1])
        lowest_low[i] = np.min(low[i-length+1:i+1])
    
    williams_r = np.full_like(high, np.nan, dtype=np.float64)
    for i in range(length-1, len(high)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50.0
    
    return williams_r

def calculate_ema(values, period):
    """Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    
    ema = np.full_like(values, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend
    wk_close = df_1w['close'].values
    ema_34_1w = calculate_ema(wk_close, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h Williams %R
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R (14) + EMA (34) + volume MA (20)
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        wr = williams_r[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: price above/below weekly EMA34
        # Need weekly close price for comparison
        wk_close_price = df_1w['close'].values
        wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
        if np.isnan(wk_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_close = wk_close_aligned[i]
        
        if position == 0:
            # Oversold with weekly uptrend: long
            if wr < -80 and weekly_close > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Overbought with weekly downtrend: short
            elif wr > -20 and weekly_close < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought or trend change
            if wr > -20 or weekly_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: oversold or trend change
            if wr < -80 or weekly_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0
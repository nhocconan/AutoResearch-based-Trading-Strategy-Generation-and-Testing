#!/usr/bin/env python3
"""
1h_4h1d_momentum_reversal_v1
Hypothesis: Use 4h momentum (price > EMA50) and 1d trend filter (price > EMA200) to establish bias,
then enter on 1h pullbacks to EMA20 with RSI oversold/overbought conditions.
Only trade during 08-20 UTC to avoid low-volume periods.
Designed for low trade frequency (15-35/year) to minimize fee drag.
Works in bull via trend filter and in bear via short side symmetry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_momentum_reversal_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for momentum filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for momentum
    close_4h = df_4h['close'].values
    ema_50_4h = calculate_ema(close_4h, 50)
    
    # Calculate 1d EMA200 for trend
    close_1d = df_1d['close'].values
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Calculate 1h EMA20 for pullback entries
    ema_20_1h = calculate_ema(close, 20)
    
    # Calculate 1h RSI for overbought/oversold
    rsi_1h = calculate_rsi(close, 14)
    
    # Align all indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_20_1h_aligned = ema_20_1h  # already on 1h
    rsi_1h_aligned = rsi_1h  # already on 1h
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_20_1h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            not in_session[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        ema_20_1h_val = ema_20_1h_aligned[i]
        rsi_val = rsi_1h_aligned[i]
        price = close[i]
        
        # Determine trend bias
        bullish_bias = price > ema_50_4h_val and price > ema_200_1d_val
        bearish_bias = price < ema_50_4h_val and price < ema_200_1d_val
        
        if position == 1:  # Long
            # Exit: RSI overbought or bias turns bearish
            if rsi_val > 70 or not bullish_bias:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: RSI oversold or bias turns bullish
            if rsi_val < 30 or not bearish_bias:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: pullback to EMA20 with RSI oversold in bullish bias
            if bullish_bias and price <= ema_20_1h_val and rsi_val < 30:
                position = 1
                signals[i] = 0.20
            # Enter short: pullback to EMA20 with RSI overbought in bearish bias
            elif bearish_bias and price >= ema_20_1h_val and rsi_val > 70:
                position = -1
                signals[i] = -0.20
    
    return signals
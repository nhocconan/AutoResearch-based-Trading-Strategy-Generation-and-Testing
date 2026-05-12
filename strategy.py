#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Filter
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) for trend direction on 1d timeframe,
# combined with RSI for momentum confirmation on 1d. Enter long when KAMA slopes up and RSI > 50,
# short when KAMA slopes down and RSI < 50. Filter trades with weekly ADX > 25 to ensure trending market.
# Designed for low frequency (10-25 trades/year) to avoid fee drag. KAMA adapts to market noise,
# reducing whipsaws in sideways markets, while RSI confirms momentum. Weekly ADX filter ensures
# we only trade in trending conditions, improving performance in both bull and bear markets.

name = "1d_KAMA_Direction_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Returns kama array.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    if n < er_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))  # |close - close[er_period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |diff| over er_period
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_rsi(close, period=14):
    """
    Calculate Relative Strength Index (RSI).
    Returns rsi array.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initial average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Subsequent averages (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Returns adx array.
    """
    n = len(high)
    adx = np.full(n, np.nan)
    if n < period * 2:
        return adx
    
    # Calculate True Range (TR)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR is just high-low
    
    # Calculate Directional Movement (DM)
    up_move = np.diff(high)
    down_move = -np.diff(low)  # Positive when low decreases
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # Initial values
    atr[period] = np.mean(tr[:period])
    plus_dm_smooth[period] = np.mean(plus_dm[:period])
    minus_dm_smooth[period] = np.mean(minus_dm[:period])
    
    # Subsequent values
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period - 1) + minus_dm[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Smooth DX to get ADX
    adx_smooth = np.full(n, np.nan)
    adx_smooth[2*period] = np.mean(dx[period:2*period])
    
    for i in range(2*period + 1, n):
        adx_smooth[i] = (adx_smooth[i-1] * (period - 1) + dx[i]) / period
    
    adx = adx_smooth
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d close
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI on 1d close
    rsi = calculate_rsi(close, period=14)
    
    # Calculate ADX on weekly data
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, period=14)
    
    # Align weekly ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA (using 2-period change)
        kama_rising = kama[i] > kama[i-2] if i >= 2 else False
        kama_falling = kama[i] < kama[i-2] if i >= 2 else False
        
        # RSI filter: >50 for bullish momentum, <50 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:
            # LONG: KAMA rising AND RSI > 50 AND trending market
            if kama_rising and rsi_bullish and trending:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling AND RSI < 50 AND trending market
            elif kama_falling and rsi_bearish and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 50 OR market not trending
            if not (kama_rising and rsi_bullish and trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 50 OR market not trending
            if not (kama_falling and rsi_bearish and trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
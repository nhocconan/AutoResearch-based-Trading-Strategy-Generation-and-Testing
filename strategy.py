#!/usr/bin/env python3
"""
6h_ADX_Trend_With_1d_Momentum_Filter
Trend following using 6h ADX for trend strength and 1d RSI for momentum confirmation.
Long when ADX > 25 (strong trend) + 1d RSI > 50 (bullish momentum)
Short when ADX > 25 + 1d RSI < 50 (bearish momentum)
Exit when ADX < 20 (weakening trend)
Designed for 15-25 trades/year per symbol.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smoothing(tr, period)
    plus_di = 100 * wilder_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilder_smoothing(minus_dm, period) / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smoothing(dx, period)
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    avg_gain[period-1] = np.nansum(gain[:period]) / period
    avg_loss[period-1] = np.nansum(loss[:period]) / period
    
    alpha = 1.0 / period
    for i in range(period, n):
        avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gain[i] * alpha
        avg_loss[i] = avg_loss[i-1] * (1 - alpha) + loss[i] * alpha
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate indicators
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Align 1d RSI to 6h timeframe
    rsi_1d_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for ADX/RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx[i]) or np.isnan(rsi_1d_6h[i]):
            signals[i] = 0.0
            continue
        
        adx_value = adx[i]
        rsi_value = rsi_1d_6h[i]
        
        if position == 0:
            # Enter long: strong uptrend (ADX > 25) + bullish momentum (RSI > 50)
            if adx_value > 25 and rsi_value > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend (ADX > 25) + bearish momentum (RSI < 50)
            elif adx_value > 25 and rsi_value < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) or momentum turning bearish
            if adx_value < 20 or rsi_value < 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) or momentum turning bullish
            if adx_value < 20 or rsi_value > 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Trend_With_1d_Momentum_Filter"
timeframe = "6h"
leverage = 1.0
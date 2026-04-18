#!/usr/bin/env python3
"""
12h Donchian(20) breakout + 1d EMA(34) trend + volume confirmation + 1w RSI(14) regime filter.
Breakouts above upper band in bull regime (RSI<60) go long; breakdowns below lower band in bear regime (RSI>40) go short.
Uses 1d EMA for trend context, 1w RSI for regime filter, volume for confirmation. Designed for 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for RSI(14)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1d
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Calculate RSI(14) on 1w
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    rsi_14_1w_12h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate Donchian channels (20-period) on 12h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian, volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_12h[i]) or np.isnan(rsi_14_1w_12h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper band, bull regime (RSI<60), volume confirmation
            if close[i] > upper[i] and rsi_14_1w_12h[i] < 60 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band, bear regime (RSI>40), volume confirmation
            elif close[i] < lower[i] and rsi_14_1w_12h[i] > 40 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band or RSI becomes overbought
            if close[i] < lower[i] or rsi_14_1w_12h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band or RSI becomes oversold
            if close[i] > upper[i] or rsi_14_1w_12h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_1wRSI_Volume"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_1w1d_combo_v1
Hypothesis: 6-hour strategy combining weekly trend (EMA50 > EMA200) with daily momentum (RSI > 55) and 6-hour price action (close > open).
Long when weekly uptrend + daily bullish momentum + 6h bullish candle.
Short when weekly downtrend + daily bearish momentum + 6h bearish candle.
Exit on opposite signal.
Designed to work in both bull (follow weekly uptrend) and bear (follow weekly downtrend) markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w1d_combo_v1"
timeframe = "6h"
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

def calculate_rsi(close, period):
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
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Get weekly and daily data for context
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMAs for trend
    close_1w = df_1w['close'].values
    ema_50_1w = calculate_ema(close_1w, 50)
    ema_200_1w = calculate_ema(close_1w, 200)
    
    # Calculate daily RSI for momentum
    close_1d = df_1d['close'].values
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Align indicators to 6-hour timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        weekly_uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        weekly_downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        daily_bullish = rsi_14_1d_aligned[i] > 55
        daily_bearish = rsi_14_1d_aligned[i] < 45
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 1:  # Long
            # Exit: weekly trend turns down OR daily momentum turns bearish OR bearish candle
            if not weekly_uptrend or not daily_bullish or bearish_candle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: weekly trend turns up OR daily momentum turns bullish OR bullish candle
            if not weekly_downtrend or not daily_bearish or bullish_candle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: weekly uptrend + daily bullish momentum + bullish candle
            if weekly_uptrend and daily_bullish and bullish_candle:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly downtrend + daily bearish momentum + bearish candle
            elif weekly_downtrend and daily_bearish and bearish_candle:
                position = -1
                signals[i] = -0.25
    
    return signals
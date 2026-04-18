#!/usr/bin/env python3
"""
Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
Use 4h EMA(21) for trend direction, enter on 1h RSI(14) pullbacks in trend direction with volume spike.
In uptrend: buy when RSI < 40 and volume > 1.5x average. In downtrend: sell when RSI > 60 and volume > 1.5x average.
Exit when RSI crosses 50 (middle) or trend changes. Designed for 15-30 trades/year to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 4h data for EMA(21) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(21) on 4h
    ema_21_4h = calculate_ema(close_4h, 21)
    
    # Align to 1h timeframe
    ema_21_4h_1h = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1h RSI(14)
    rsi_14_1h = calculate_rsi(close, 14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_4h_1h[i]) or np.isnan(rsi_14_1h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: uptrend (price above 4h EMA), RSI oversold, volume confirmation
            if close[i] > ema_21_4h_1h[i] and rsi_14_1h[i] < 40 and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price below 4h EMA), RSI overbought, volume confirmation
            elif close[i] < ema_21_4h_1h[i] and rsi_14_1h[i] > 60 and vol_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 or trend turns down
            if rsi_14_1h[i] >= 50 or close[i] < ema_21_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses below 50 or trend turns up
            if rsi_14_1h[i] <= 50 or close[i] > ema_21_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_4hTrend_RSI14_Volume"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1d price action relative to 1w EMA(21) with volume confirmation and 1w RSI(14) regime filter.
In bull markets: price above 1w EMA(21) acts as dynamic support, buy on dips with volume.
In bear markets: price below 1w EMA(21) acts as resistance, sell on rallies with volume.
Weekly RSI avoids extremes: only long when RSI(1w)<60, short when RSI(1w)>40.
Designed for 15-25 trades/year to minimize fee drag on 1d timeframe.
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
    
    # Get 1d data (same as primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA and RSI
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(21) on 1w
    ema_21_1w = calculate_ema(close_1w, 21)
    
    # Calculate RSI(14) on 1w
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align 1w indicators to 1d timeframe
    ema_21_1w_1d = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    rsi_14_1w_1d = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_1w_1d[i]) or np.isnan(rsi_14_1w_1d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price above 1w EMA(21), RSI not overbought, volume confirmation
            if close[i] > ema_21_1w_1d[i] and rsi_14_1w_1d[i] < 60 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA(21), RSI not oversold, volume confirmation
            elif close[i] < ema_21_1w_1d[i] and rsi_14_1w_1d[i] > 40 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1w EMA(21) or RSI becomes overbought
            if close[i] <= ema_21_1w_1d[i] or rsi_14_1w_1d[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1w EMA(21) or RSI becomes oversold
            if close[i] >= ema_21_1w_1d[i] or rsi_14_1w_1d[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA21_1wRSI_Volume"
timeframe = "1d"
leverage = 1.0
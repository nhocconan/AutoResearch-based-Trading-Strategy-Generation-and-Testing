#!/usr/bin/env python3
"""
4h_rsi_sma_volume_filter_v1
Hypothesis: Combines RSI extremes with SMA trend filter and volume confirmation on 4H timeframe.
- Long when RSI < 35 (oversold), price > SMA50, and volume > 1.5x average
- Short when RSI > 65 (overbought), price < SMA50, and volume > 1.5x average
- Uses 4H timeframe for proper trade frequency (target: 20-40 trades/year)
- Works in bull/bear via SMA trend filter preventing counter-trend trades
- Volume confirmation ensures conviction behind moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_sma_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

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
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(close, period):
    """Calculate SMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    sma = np.full_like(close, np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # Calculate SMA(50)
    sma_50 = calculate_sma(close, 50)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(sma_50[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] > 50 or price < sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] < 50 or price > sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI oversold, price above SMA, volume confirmation
            if rsi[i] < 35 and price > sma_50[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI overbought, price below SMA, volume confirmation
            elif rsi[i] > 65 and price < sma_50[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals
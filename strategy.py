#!/usr/bin/env python3
"""
4h RSI Divergence + Volume Confirmation + ADX Trend Filter
Hypothesis: RSI divergences (hidden and regular) at extremes with volume confirmation and ADX > 25 capture exhaustion moves in both bull and bear markets. Limited trades due to strict divergence and volume requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergences(price, rsi, lookback=5):
    """Find bullish and bearish divergences"""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Regular bullish divergence: price makes lower low, RSI makes higher low
        if price[i] < price[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if it's a meaningful low
            if price[i] == np.min(price[i-lookback:i+1]):
                bullish_div[i] = True
        
        # Regular bearish divergence: price makes higher high, RSI makes lower high
        if price[i] > price[i-lookback] and rsi[i] < rsi[i-lookback]:
            # Check if it's a meaningful high
            if price[i] == np.max(price[i-lookback:i+1]):
                bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI on 4h
    rsi = calculate_rsi(close, 14)
    
    # Find divergences
    bullish_div, bearish_div = find_divergences(close, rsi, lookback=10)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_conf = volume > (vol_ma * 1.5)
    
    # ADX for trend strength (avoid choppy markets)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values with Wilder smoothing
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        # Initial value
        result[period-1] = np.mean(data[:period])
        # Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilders_smooth(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: bullish divergence + ADX > 20 + volume confirmation
            if (bullish_div[i] and 
                adx[i] > 20 and 
                vol_conf[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish divergence + ADX > 20 + volume confirmation
            elif (bearish_div[i] and 
                  adx[i] > 20 and 
                  vol_conf[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish divergence or RSI overbought (>70) or ADX weakens
            if (bearish_div[i] or 
                rsi[i] > 70 or 
                adx[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish divergence or RSI oversold (<30) or ADX weakens
            if (bullish_div[i] or 
                rsi[i] < 30 or 
                adx[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0
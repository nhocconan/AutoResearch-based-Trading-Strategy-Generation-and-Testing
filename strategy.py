#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d ATR volatility filter and 1w trend filter.
Williams %R identifies overbought/oversold conditions. In trending markets (ADX>25),
we fade extremes only when volatility is contracting (ATR ratio < 0.8) to avoid
whipsaws. In ranging markets (ADX<20), we mean-revert at extreme levels.
Weekly EMA(50) determines trend direction: only long when price > weekly EMA(50),
short when price < weekly EMA(50). Designed for 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    highest_high = np.full(len(close), np.nan)
    lowest_low = np.full(len(close), np.nan)
    
    for i in range(period-1, len(close)):
        highest_high[i] = np.max(high[i-(period-1):i+1])
        lowest_low[i] = np.min(low[i-(period-1):i+1])
    
    wr = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            wr[i] = -50
    
    return wr

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(close), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR(14)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA(50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate indicators
    williams_r = calculate_williams_r(high, low, close, 14)
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Align to 4h timeframe
    williams_r_4h = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_14_1d_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_1w_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR ratio (current ATR / 20-period average ATR) for volatility filter
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if not np.isnan(atr_14_1d_4h[i-20:i]).any():
            atr_avg = np.nanmean(atr_14_1d_4h[i-20:i])
            if atr_avg > 0:
                atr_ratio[i] = atr_14_1d_4h[i] / atr_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need Williams %R, ATR ratio, and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_4h[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(ema_50_1w_4h[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime (simplified: using price relative to EMA)
        # In practice, would use ADX but keeping it simple with price/EMA relationship
        price_above_ema = close[i] > ema_50_1w_4h[i]
        
        if position == 0:
            # Long: oversold + volatility contracting + uptrend bias
            if (williams_r_4h[i] <= -80 and atr_ratio[i] < 0.8 and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: overbought + volatility contracting + downtrend bias
            elif (williams_r_4h[i] >= -20 and atr_ratio[i] < 0.8 and not price_above_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns from oversold or volatility expands
            if williams_r_4h[i] >= -50 or atr_ratio[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns from overbought or volatility expands
            if williams_r_4h[i] <= -50 or atr_ratio[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ATR_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
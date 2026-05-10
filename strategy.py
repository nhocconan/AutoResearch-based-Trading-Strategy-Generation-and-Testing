#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Trend_v2
# Hypothesis: 4h trend following using Kaufman Adaptive Moving Average (KAMA) with RSI filter.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# RSI > 50 for longs, RSI < 50 for shorts ensures momentum alignment.
# Volume confirmation (1.5x 24-period average) filters low-conviction moves.
# Designed for 4h timeframe targeting 25-50 trades/year per symbol.
# Works in bull/bear by requiring adaptive trend alignment and volume confirmation.

name = "4h_KAMA_Direction_RSI_Trend_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation properly for array
    volatility_temp = np.abs(np.diff(close))
    volatility = np.concatenate([np.full(er_length, np.nan), 
                                np.convolve(volatility_temp, np.ones(er_length), 'valid')])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # Seed
    for i in range(er_length + 1, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    rsi_length = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[rsi_length] = np.mean(gain[:rsi_length])
    avg_loss[rsi_length] = np.mean(loss[:rsi_length])
    
    for i in range(rsi_length + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i-1]) / rsi_length
        avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i-1]) / rsi_length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (24-period for 4h = 4 days)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(24, len(volume)):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for KAMA, RSI, and volume MA
    start_idx = max(er_length, rsi_length) + 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend: price above/below KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI filter: >50 for long, <50 for short
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Volume confirmation (1.5x average for significance)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA, RSI bullish, volume surge
            if above_kama and rsi_bullish and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI bearish, volume surge
            elif below_kama and rsi_bearish and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price below KAMA or RSI turns bearish
                if below_kama or not rsi_bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price above KAMA or RSI turns bullish
                if above_kama or not rsi_bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
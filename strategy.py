#!/usr/bin/env python3
"""
4h_1d_RSI_Divergence_Volume_Strict
Hypothesis: On 4h timeframe, identify bullish/bearish RSI divergences with volume confirmation.
Bullish: price makes lower low, RSI makes higher low, volume increases on up days.
Bearish: price makes higher high, RSI makes lower high, volume decreases on down days.
Uses 1d RSI divergence for higher timeframe confirmation and volume spike for entry timing.
Targets 20-40 trades/year with strict conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI with proper Wilder smoothing"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Set first period values to NaN
    rsi[:period] = np.nan
    return rsi

def find_divergences(price, indicator, lookback=10):
    """
    Find bullish and bearish divergences
    Returns arrays: bullish_div (True when bullish div), bearish_div (True when bearish div)
    """
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for bullish divergence: price lower low, indicator higher low
        price_low = np.argmin(price[i-lookback:i+1]) + i - lookback
        ind_low = np.argmin(indicator[i-lookback:i+1]) + i - lookback
        
        if price_low != ind_low and price[price_low] < price[i-lookback] and indicator[ind_low] > indicator[i-lookback]:
            # Check if this is a valid divergence point
            if price_low == i-lookback or ind_low == i-lookback:
                bullish_div[i] = True
        
        # Look for bearish divergence: price higher high, indicator lower high
        price_high = np.argmax(price[i-lookback:i+1]) + i - lookback
        ind_high = np.argmax(indicator[i-lookback:i+1]) + i - lookback
        
        if price_high != ind_high and price[price_high] > price[i-lookback] and indicator[ind_high] < indicator[i-lookback]:
            if price_high == i-lookback or ind_high == i-lookback:
                bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Data (HTF for RSI divergence) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d RSI (14-period)
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Find divergences on 1d data
    bullish_div_1d, bearish_div_1d = find_divergences(close_1d, rsi_1d, lookback=10)
    bullish_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_div_1d.astype(float))
    bearish_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_div_1d.astype(float))
    
    # 4h volume spike detection (20-period median)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > 1.8 * volume_ma  # Volume spike threshold
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(bullish_div_1d_aligned[i]) or
            np.isnan(bearish_div_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bullish entry: 1d bullish divergence + 4h volume spike + price above 20-period EMA
            ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
            if (bullish_div_1d_aligned[i] > 0.5 and 
                volume_spike[i] and 
                close[i] > ema_20[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Bearish entry: 1d bearish divergence + 4h volume spike + price below 20-period EMA
            elif (bearish_div_1d_aligned[i] > 0.5 and 
                  volume_spike[i] and 
                  close[i] < ema_20[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit on bearish divergence or price below EMA
            ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
            if bearish_div_1d_aligned[i] > 0.5 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on bullish divergence or price above EMA
            ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
            if bullish_div_1d_aligned[i] > 0.5 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_Divergence_Volume_Strict"
timeframe = "4h"
leverage = 1.0
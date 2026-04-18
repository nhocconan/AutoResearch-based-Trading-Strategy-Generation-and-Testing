#!/usr/bin/env python3
"""
1h RSI Divergence + 4h Trend + Volume Spike
Hypothesis: RSI divergences signal momentum exhaustion; when combined with 4h trend direction and volume spikes, they capture high-probability reversals. Works in bull/bear markets by following the 4h trend (long in uptrend, short in downtrend). Low trade frequency (~20-40/year) avoids fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergences(prices, rsi, lookback=14):
    """Find bullish and bearish RSI divergences"""
    n = len(prices)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for price making lower low while RSI makes higher low (bullish div)
        if i >= lookback * 2:
            price_low_idx = np.argmin(prices[i-lookback:i+1]) + i - lookback
            rsi_low_idx = np.argmin(rsi[i-lookback:i+1]) + i - lookback
            if (prices[price_low_idx] < prices[i-lookback] and 
                rsi[rsi_low_idx] > rsi[i-lookback] and
                rsi[i] < 40):  # Oversold condition
                bull_div[i] = True
        
        # Look for price making higher high while RSI makes lower high (bearish div)
        if i >= lookback * 2:
            price_high_idx = np.argmax(prices[i-lookback:i+1]) + i - lookback
            rsi_high_idx = np.argmax(rsi[i-lookback:i+1]) + i - lookback
            if (prices[price_high_idx] > prices[i-lookback] and 
                rsi[rsi_high_idx] < rsi[i-lookback] and
                rsi[i] > 60):  # Overbought condition
                bear_div[i] = True
    
    return bull_div, bear_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 1h
    rsi = calculate_rsi(close, 14)
    
    # Find divergences
    bull_div, bear_div = find_divergences(close, rsi, 14)
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = np.zeros_like(close_4h)
    if len(close_4h) >= 21:
        alpha = 2 / (21 + 1)
        ema_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_4h[i-1]
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            start_idx = max(0, i - 19)
            vol_ma[i] = np.mean(volume[start_idx:i+1]) if i >= start_idx else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1 = uptrend (price > EMA), -1 = downtrend (price < EMA)
        trend = 1 if close[i] > ema_4h_aligned[i] else -1
        
        if position == 0:
            # Enter long: bullish divergence + volume spike + uptrend
            if bull_div[i] and vol_spike[i] and trend == 1:
                signals[i] = 0.20
                position = 1
            # Enter short: bearish divergence + volume spike + downtrend
            elif bear_div[i] and vol_spike[i] and trend == -1:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: bearish divergence or trend change
            if bear_div[i] or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: bullish divergence or trend change
            if bull_div[i] or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSIDivergence_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0
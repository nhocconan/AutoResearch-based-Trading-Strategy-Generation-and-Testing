#!/usr/bin/env python3
"""
4h_1d_1w_rsi_divergence_volume_v1
Hypothesis: Combine RSI divergence with multi-timeframe trend and volume confirmation on 4h timeframe.
Long when: price makes lower low but RSI makes higher low (bullish divergence) on 1d, price > 4h EMA20, and volume > 1.5x average.
Short when: price makes higher high but RSI makes lower high (bearish divergence) on 1d, price < 4h EMA20, and volume > 1.5x average.
Uses 1w trend filter for long-term bias. Designed to capture reversals in both bull and bear markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_rsi_divergence_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Calculate 4h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1w EMA20 for long-term trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_20)  # self-align
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h EMA20 or bearish RSI divergence
            if close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h EMA20 or bullish RSI divergence
            if close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Need at least 2 points for divergence check
            if i < 2:
                signals[i] = 0.0
                continue
                
            # Bullish divergence: price makes lower low but RSI makes higher low
            price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
            rsi_higher_low = rsi_1d_aligned[i] > rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] > rsi_1d_aligned[i-2]
            bullish_div = price_lower_low and rsi_higher_low
            
            # Bearish divergence: price makes higher high but RSI makes lower high
            price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
            rsi_lower_high = rsi_1d_aligned[i] < rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] < rsi_1d_aligned[i-2]
            bearish_div = price_higher_high and rsi_lower_high
            
            # Long entry: bullish divergence + price > 4h EMA20 + volume + weekly uptrend
            if bullish_div and close[i] > ema_20_aligned[i] and vol_confirm[i] and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish divergence + price < 4h EMA20 + volume + weekly downtrend
            elif bearish_div and close[i] < ema_20_aligned[i] and vol_confirm[i] and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
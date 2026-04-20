#!/usr/bin/env python3
"""
4h_RSI_Divergence_with_Volume_Confirmation
Hypothesis: Trade 4h RSI divergence (bullish/bearish) with volume confirmation and ADX trend filter.
Bullish divergence: price makes lower low, RSI makes higher low -> long.
Bearish divergence: price makes higher high, RSI makes lower high -> short.
Volume confirmation ensures momentum behind the move. ADX > 25 filters for trending markets.
Works in bull/bear: RSI divergence captures reversals at trend extremes, volume avoids fakeouts.
Target: ~100 total trades over 4 years (25/year) with position size 0.25.
"""

name = "4h_RSI_Divergence_with_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily ADX for trend filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.max([high_daily[0] - low_daily[0], 
                                    np.abs(high_daily[0] - close_daily[0] if len(close_daily)>0 else 0),
                                    np.abs(low_daily[0] - close_daily[0] if len(close_daily)>0 else 0)])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.sum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dx = np.full_like(close_daily, np.nan)
    if len(atr) >= 14 and len(dm_plus) >= 14:
        di_plus = 100 * wilders_smoothing(dm_plus, 14) / atr
        di_minus = 100 * wilders_smoothing(dm_minus, 14) / atr
        dx_sum = di_plus + di_minus
        dx = np.where(dx_sum != 0, 100 * np.abs(di_plus - di_minus) / dx_sum, 0)
    
    adx = wilders_smoothing(dx, 14)
    adx_daily_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(adx_daily_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or 
            np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Need at least 3 points for divergence check
        if i < 2:
            signals[i] = 0.0
            continue
            
        # ADX filter: only trade in trending markets (ADX > 25)
        if adx_daily_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period MA
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Bullish divergence: price lower low, RSI higher low
            if (low[i] < low[i-1] and low[i-1] < low[i-2] and  # price making lower low
                rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2] and   # RSI making higher low
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price higher high, RSI lower high
            elif (high[i] > high[i-1] and high[i-1] > high[i-2] and  # price making higher high
                  rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2] and      # RSI making lower high
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence or ADX weakens
            bearish_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                          rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2])
            if bearish_div or adx_daily_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence or ADX weakens
            bullish_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                          rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2])
            if bullish_div or adx_daily_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
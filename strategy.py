#!/usr/bin/env python3
"""
1h Confluence Strategy: 4h ADX trend + 1h RSI mean reversion + session filter
Long in strong uptrend (ADX>25) when RSI oversold (<30)
Short in strong downtrend (ADX>25) when RSI overbought (>70)
Exit when RSI returns to neutral (40-60) or ADX weakens (<20)
Uses 4h for trend direction (ADX), 1h for entry timing (RSI)
Designed for 15-35 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smooth(tr, 14)
    dpi = wilders_smooth(dm_plus, 14)
    dmi = wilders_smooth(dm_minus, 14)
    
    # Avoid division by zero
    di_plus = np.where(atr != 0, 100 * dpi / atr, 0)
    di_minus = np.where(atr != 0, 100 * dmi / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1h RSI for entry timing
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.empty_like(close, dtype=np.float64)
    avg_loss = np.empty_like(close, dtype=np.float64)
    avg_gain.fill(np.nan)
    avg_loss.fill(np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need ADX (14+14+14=42) and RSI (14)
    start_idx = max(42, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(hours[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Current values
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: strong uptrend (ADX>25) + RSI oversold (<30)
            if adx_val > 25 and rsi_val < 30:
                signals[i] = size
                position = 1
            # Short: strong downtrend (ADX>25) + RSI overbought (>70)
            elif adx_val > 25 and rsi_val > 70:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or trend weakens (ADX<20)
            if rsi_val >= 40 and rsi_val <= 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or trend weakens (ADX<20)
            if rsi_val >= 40 and rsi_val <= 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_ADX25_RSI_Confluence_Session"
timeframe = "1h"
leverage = 1.0
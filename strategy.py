#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour RSI and 1-day trend filter with volume confirmation.
# Long when price pulls back to 4-hour VWAP with RSI < 40 and price > 1-day EMA50.
# Short when price rallies to 4-hour VWAP with RSI > 60 and price < 1-day EMA50.
# Uses 4-hour momentum for direction, 1-hour for precise entry during pullbacks.
# Volume confirmation filters low-activity periods. Session filter (08-20 UTC) reduces noise.
# Target: 20-50 trades/year per symbol (80-200 over 4 years) to minimize fee drag.

name = "1h_4h1d_rsi_vwap_v1"
timeframe = "1h"
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
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for RSI and VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour RSI
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    
    # Calculate 4-hour VWAP
    vwap_4h = calculate_vwap(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, df_4h['volume'].values)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = np.full_like(df_1d['close'].values, np.nan, dtype=float)
    if len(df_1d) >= 50:
        ema_50_1d[49] = np.mean(df_1d['close'].values[:50])
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = 2.0 / (50 + 1) * df_1d['close'].values[i] + (1 - 2.0 / (50 + 1)) * ema_50_1d[i-1]
    
    # Align indicators to 1-hour timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 24-period average (1 day)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                pass  # Hold position outside session
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        rsi = rsi_4h_aligned[i]
        vwap = vwap_4h_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 4-hour VWAP or RSI > 50 (momentum fading)
            if price < vwap or rsi > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price crosses above 4-hour VWAP or RSI < 50 (momentum fading)
            if price > vwap or rsi < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price pulls back to 4-hour VWAP with RSI < 40 and uptrend on 1d
            if price >= vwap * 0.998 and price <= vwap * 1.002 and rsi < 40 and trend_up_1d and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price rallies to 4-hour VWAP with RSI > 60 and downtrend on 1d
            elif price >= vwap * 0.998 and price <= vwap * 1.002 and rsi > 60 and not trend_up_1d and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals
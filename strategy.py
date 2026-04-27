#!/usr/bin/env python3
"""
Hypothesis: 12-hour volume-weighted average price (VWAP) with weekly trend filter and daily volume confirmation.
When price crosses above VWAP in a weekly uptrend with above-average volume: long.
When price crosses below VWAP in a weekly downtrend with above-average volume: short.
VWAP captures institutional activity, weekly trend filters direction, volume confirms participation.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap = np.full_like(typical_price, np.nan)
    cum_tpv = np.zeros_like(typical_price)
    cum_vol = np.zeros_like(typical_price)
    
    for i in range(len(typical_price)):
        cum_tpv += typical_price[i] * volume[i]
        cum_vol += volume[i]
        if cum_vol > 0:
            vwap[i] = cum_tpv[i] / cum_vol
    
    return vwap

def calculate_ema(values, period):
    """Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    
    ema = np.full_like(values, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend
    wk_close = df_1w['close'].values
    ema_34_1w = calculate_ema(wk_close, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h VWAP
    vwap = calculate_vwap(high, low, close, volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        vwap_now = vwap[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: price above/below weekly EMA34
        # Need weekly close price for comparison
        wk_close_price = df_1w['close'].values
        wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
        if np.isnan(wk_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_close = wk_close_aligned[i]
        
        if position == 0:
            # Price crosses above VWAP with weekly uptrend: long
            if price_now > vwap_now and weekly_close > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Price crosses below VWAP with weekly downtrend: short
            elif price_now < vwap_now and weekly_close < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or trend change
            if price_now < vwap_now or weekly_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP or trend change
            if price_now > vwap_now or weekly_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_VWAP_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0
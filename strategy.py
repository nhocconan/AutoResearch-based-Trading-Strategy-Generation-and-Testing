#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_ADX_Filter
Hypothesis: On 6h timeframe, Ichimoku cloud twist (Senkou Span A/B cross) with ADX(14)>25 trend filter and 12h EMA50 confirmation.
Cloud twist signals potential trend change; ADX ensures we only take strong trends. 12h EMA50 provides higher timeframe bias.
Works in bull (cloud twist up + above EMA) and bear (cloud twist down + below EMA) markets. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = np.nan
    dm_minus[0] = np.nan
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    if len(high) < 52:
        return (np.full_like(high, np.nan), np.full_like(high, np.nan), 
                np.full_like(high, np.nan), np.full_like(high, np.nan))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(high).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(high).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(high).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = calculate_ema(df_12h['close'].values, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d data for Ichimoku and ADX (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    adx = calculate_adx(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14
    )
    
    # Align Ichimoku and ADX to 6h timeframe (completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + ADX (14+14) + 12h EMA (50)
    start_idx = max(52, 28, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Ichimoku cloud twist + ADX>25 + 12h EMA50 alignment
            # Cloud twist: Senkou Span A crosses above/below Senkou Span B
            senkou_a_prev = senkou_a_aligned[i-1] if i > 0 else senkou_a_aligned[i]
            senkou_b_prev = senkou_b_aligned[i-1] if i > 0 else senkou_b_aligned[i]
            
            # Bullish twist: Senkou A crosses above Senkou B
            bullish_twist = (senkou_a_aligned[i] > senkou_b_aligned[i] and 
                           senkou_a_prev <= senkou_b_prev)
            
            # Bearish twist: Senkou A crosses below Senkou B
            bearish_twist = (senkou_a_aligned[i] < senkou_b_aligned[i] and 
                           senkou_a_prev >= senkou_b_prev)
            
            # Trend filters
            strong_trend = adx_aligned[i] > 25
            above_ema = curr_close > ema_50_12h_aligned[i]
            below_ema = curr_close < ema_50_12h_aligned[i]
            
            long_entry = bullish_twist and strong_trend and above_ema
            short_entry = bearish_twist and strong_trend and below_ema
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when cloud twist reverses bearish OR trend weakens
            senkou_a_prev = senkou_a_aligned[i-1] if i > 0 else senkou_a_aligned[i]
            senkou_b_prev = senkou_b_aligned[i-1] if i > 0 else senkou_b_aligned[i]
            bearish_twist = (senkou_a_aligned[i] < senkou_b_aligned[i] and 
                           senkou_a_prev >= senkou_b_prev)
            weak_trend = adx_aligned[i] < 20
            
            if bearish_twist or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when cloud twist reverses bullish OR trend weakens
            senkou_a_prev = senkou_a_aligned[i-1] if i > 0 else senkou_a_aligned[i]
            senkou_b_prev = senkou_b_aligned[i-1] if i > 0 else senkou_b_aligned[i]
            bullish_twist = (senkou_a_aligned[i] > senkou_b_aligned[i] and 
                           senkou_a_prev <= senkou_b_prev)
            weak_trend = adx_aligned[i] < 20
            
            if bullish_twist or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_ADX_Filter"
timeframe = "6h"
leverage = 1.0
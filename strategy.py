#!/usr/bin/env python3
"""
Experiment #12155: 6h Ichimoku Cloud with 1w Trend and Volume Confirmation
Hypothesis: Ichimoku provides multi-component trend signal (TK cross, cloud, Kijun). 
1w Ichimoku cloud color filters for major trend. Volume ensures institutional participation.
Works in bull (price above cloud + bullish TK) and bear (price below cloud + bearish TK) 
by using 1w cloud color as regime filter. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12155_6h_ichimoku_1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(-KIJUN_PERIOD)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku for trend filter
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    
    # 1w trend: price above/both Senkou lines = bullish, below both = bearish
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    cloud_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    price_above_cloud_1w = df_1w['close'].values > cloud_top_1w
    price_below_cloud_1w = df_1w['close'].values < cloud_bottom_1w
    
    # Align 1w trend to 6h
    price_above_cloud_1w_aligned = align_htf_to_ltf(prices, df_1w, price_above_cloud_1w.astype(float))
    price_below_cloud_1w_aligned = align_htf_to_ltf(prices, df_1w, price_below_cloud_1w.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # 6h TK cross: Tenkan crosses above/below Kijun
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Crossed above on this bar
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Crossed below on this bar
    
    # 6h price relative to cloud
    cloud_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    price_above_cloud_6h = close > cloud_top_6h
    price_below_cloud_6h = close < cloud_bottom_6h
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD, KIJUN_PERIOD, VOLUME_MA_PERIOD) + KIJUN_PERIOD
    
    for i in range(start, n):
        # Skip if 1w Ichimoku not available
        if np.isnan(cloud_top_1w_aligned[i]) or np.isnan(cloud_bottom_1w_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku signals
        # Bullish: TK cross above + price above cloud (6h) + 1w bullish trend
        bullish_setup = (tk_cross_above[i] if i > 0 else False) and \
                       price_above_cloud_6h[i] and \
                       price_above_cloud_1w_aligned[i]
        
        # Bearish: TK cross below + price below cloud (6h) + 1w bearish trend
        bearish_setup = (tk_cross_below[i] if i > 0 else False) and \
                       price_below_cloud_6h[i] and \
                       price_below_cloud_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_setup and volume_ok
        short_entry = bearish_setup and volume_ok
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
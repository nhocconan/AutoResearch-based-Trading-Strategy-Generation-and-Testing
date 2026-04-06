#!/usr/bin/env python3
"""
13739 - 6h Ichimoku Cloud with 1d Trend Filter & Volume Confirmation
Hypothesis: Ichimoku (TK Cross + Cloud) on 6h provides reliable entry signals,
while 1d trend (via Kumo twist) filters direction and volume confirms momentum.
Works in bull/bear: Kumo acts as dynamic S/R, TK crosses catch reversals.
Target: 50-150 trades over 4 years (12-37/year).
"""

name = "exp_13739_6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Ichimoku parameters (standard)
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
# 1d trend: Kumo twist (Senkou A vs B)
# Volume confirmation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h Ichimoku
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high_6h, low_6h, close_6h)
    
    # Align Ichimoku components to 6s timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate 1d Kumo twist for trend filter (Senkou A > Senkou B = bullish)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate Senkou Span A and B on 1d
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                    pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2)
    # Kumo twist: Senkou A > Senkou B = bullish twist
    kumo_twist_bullish = (senkou_a_1d > senkou_b_1d).astype(float)
    kumo_twist_6h = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish)
    
    # ATR for stop loss (using 6h data)
    atr = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    # Volume MA for 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need enough data for Ichimoku)
    start = max(SENKOU_B_PERIOD + KIJUN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(kumo_twist_6h[i]) or np.isnan(volume_ma_6h[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            if close_6h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_6h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume_6h[i] > (volume_ma_6h[i] * VOLUME_THRESHOLD)
        
        # Ichimoku signals
        price = close_6h[i]
        tenkan_i = tenkan_6h[i]
        kijun_i = kijun_6h[i]
        senkou_a_i = senkou_a_6h[i]
        senkou_b_i = senkou_b_6h[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_i, senkou_b_i)
        cloud_bottom = min(senkou_a_i, senkou_b_i)
        
        # TK Cross signals
        tk_cross_bull = tenkan_i > kijun_i
        tk_cross_bear = tenkan_i < kijun_i
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Long: Price above cloud + TK cross bull + bullish Kumo twist (1d) + volume
        long_signal = (price_above_cloud and tk_cross_bull and 
                       kumo_twist_6h[i] > 0.5 and volume_ok)
        
        # Short: Price below cloud + TK cross bear + bearish Kumo twist (1d) + volume
        short_signal = (price_below_cloud and tk_cross_bear and 
                        kumo_twist_6h[i] < 0.5 and volume_ok)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = price
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = price
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below cloud OR TK cross bear
            exit_signal = (price < cloud_top) or (tk_cross_bear and price < kijun_i)
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: Price crosses above cloud OR TK cross bull
            exit_signal = (price > cloud_bottom) or (tk_cross_bull and price > kijun_i)
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals
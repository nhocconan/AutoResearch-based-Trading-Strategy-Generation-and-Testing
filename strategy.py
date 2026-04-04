#!/usr/bin/env python3
"""
exp_6755_6h_ichimoku_cloud_tk_cross_v1
Hypothesis: 6h Ichimoku system with TK cross + cloud filter from 1d timeframe.
Long when price > cloud (bullish) and Tenkan crosses above Kijun.
Short when price < cloud (bearish) and Tenkan crosses below Kijun.
Cloud from 1d provides higher timeframe structure to avoid whipsaws.
TK cross on 6h provides timely entries within the trend.
Designed for 6h timeframe to capture medium swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by aligning with 1d cloud direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6755_6h_ichimoku_cloud_tk_cross_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
# Ichimoku parameters (standard)
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
# TK cross requires both lines
# Cloud: Senkou Span A (leading span 1) and Senkou Span B (leading span 2)
# Signal size
SIGNAL_SIZE = 0.25
# ATR for stoploss
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
# Maximum hold bars (~10 days on 6h = 40 bars)
MAX_HOLD_BARS = 40

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # need enough for Senkou B period
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward KIJUN_PERIOD
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(KIJUN_PERIOD)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_B_PERIOD shifted forward KIJUN_PERIOD
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                    pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    # Chikou Span (Lagging Span): close shifted back KIJUN_PERIOD (not used for signals)
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Calculate LTF (6h) indicators for TK cross
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen and Kijun-sen on 6h
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period - need enough for all indicators
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD) + KIJUN_PERIOD + 10
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available (NaN from alignment)
        if np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Ichimoku cloud: price above/both Senkou lines
        # Cloud top is max(Senkou A, Senkou B), bottom is min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Price above cloud = bullish, below cloud = bearish
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross on 6h: Tenkan crosses Kijun
        # Need previous values to detect cross
        tenkan_prev = tenkan_6h.iloc[i-1] if i > 0 else tenkan_6h.iloc[0]
        kijun_prev = kijun_6h.iloc[i-1] if i > 0 else kijun_6h.iloc[0]
        tenkan_curr = tenkan_6h.iloc[i]
        kijun_curr = kijun_6h.iloc[i]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish_cross = tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish_cross = tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr
        
        # Entry signals: TK cross in direction of cloud
        long_signal = price_above_cloud and tk_bullish_cross
        short_signal = price_below_cloud and tk_bearish_cross
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals
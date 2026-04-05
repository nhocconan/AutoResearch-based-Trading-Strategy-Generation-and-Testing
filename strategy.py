#!/usr/bin/env python3
"""
Experiment #11431: 6h Ichimoku Cloud with 1d Kumo Twist and Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. A daily Kumo twist (Senkou A/B cross)
provides trend bias, while 6h Tenkan/Kijun cross gives precise entry. Volume confirmation ensures
momentum. Works in bull (price above cloud, bullish twist) and bear (price below cloud, bearish twist)
by using daily Kumo twist as regime filter. Target: 100-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11431_6h_ichimoku_1d_kumo_twist_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Conversion Line
KIJUN_PERIOD = 26    # Base Line
SENKOU_B_PERIOD = 52 # Leading Span B
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over last 9 periods
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over last 26 periods
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over last 52 periods shifted forward 26
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted back 26 periods (not used for signals)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

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
    
    # Load daily data ONCE before loop for Kumo twist
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku for Kumo twist detection
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    _, _, senkou_a_daily, senkou_b_daily = calculate_ichimoku(daily_high, daily_low, daily_close)
    
    # Kumo twist: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
    kumo_bullish = senkou_a_daily > senkou_b_daily
    kumo_bearish = senkou_a_daily < senkou_b_daily
    
    # Align daily Kumo twist to 6h timeframe
    kumo_bullish_aligned = align_htf_to_ltf(prices, df_daily, kumo_bullish.astype(float))
    kumo_bearish_aligned = align_htf_to_ltf(prices, df_daily, kumo_bearish.astype(float))
    
    # Calculate 6h Ichimoku components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume = prices['volume'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high_6h, low_6h, close_6h)
    
    # Calculate 6x ATR for stop loss
    atr_6h = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need enough data for Ichimoku)
    start = max(SENKOU_B_PERIOD + 26, KIJUN_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Kumo twist not available
        if np.isnan(kumo_bullish_aligned[i]) or np.isnan(kumo_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
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
        
        # Ichimoku signals
        # Price above/below cloud
        cloud_top_6h = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        price_above_cloud = close_6h[i] > cloud_top_6h
        price_below_cloud = close_6h[i] < cloud_bottom_6h
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] if i > 0 else False
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] if i > 0 else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions based on daily Kumo twist and 6h Ichimoku
        long_entry = (price_above_cloud and tk_cross_up and volume_ok and kumo_bullish_aligned[i] > 0.5)
        short_entry = (price_below_cloud and tk_cross_down and volume_ok and kumo_bearish_aligned[i] > 0.5)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_6h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_6h[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_6h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_6h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
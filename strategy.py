#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Ichimoku Cloud with 1d/1w filters for trend confirmation.
# Goes long when price breaks above Kumo cloud with Tenkan > Kijun (bullish TK cross) and 1d EMA200 filter.
# Goes short when price breaks below Kumo cloud with Tenkan < Kijun (bearish TK cross) and below 1d EMA200.
# Uses 1w trend filter to avoid counter-trend trades in strong trends.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Ichimoku provides multiple confirmation layers, reducing false breakouts.

name = "exp_13867_6h_ichimoku_1d_1w_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
EMA_1D_PERIOD = 200
EMA_1W_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo cloud (Senkou Span A/B from 26 periods ago)
    senkou_a_current = senkou_a.shift(-KUMO_SHIFT)
    senkou_b_current = senkou_b.shift(-KUMO_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a_current.values, senkou_b_current.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d and 1w data for trend filters ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_1D_PERIOD)
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_1W_PERIOD)
    
    # Align 1d and 1w EMAs to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for Ichimoku, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT, 
                EMA_1D_PERIOD, EMA_1W_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filters from 1d and 1w EMA
        above_ema_1d = close[i] > ema_1d_aligned[i]
        below_ema_1d = close[i] < ema_1d_aligned[i]
        above_ema_1w = close[i] > ema_1w_aligned[i]
        below_ema_1w = close[i] < ema_1w_aligned[i]
        
        # Ichimoku signals
        # Bullish: price above cloud, Tenkan > Kijun (TK cross bullish)
        bullish_kumo = (close[i] > senkou_a[i]) and (close[i] > senkou_b[i])
        bullish_tk = tenkan[i] > kijun[i]
        
        # Bearish: price below cloud, Tenkan < Kijun (TK cross bearish)
        bearish_kumo = (close[i] < senkou_a[i]) and (close[i] < senkou_b[i])
        bearish_tk = tenkan[i] < kijun[i]
        
        # Long signal: bullish Kumo + bullish TK + volume + above 1d/1w EMA
        long_signal = (bullish_kumo and bullish_tk and volume_ok and 
                      above_ema_1d and above_ema_1w)
        
        # Short signal: bearish Kumo + bearish TK + volume + below 1d/1w EMA
        short_signal = (bearish_kumo and bearish_tk and volume_ok and 
                       below_ema_1d and below_ema_1w)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish TK cross or price below cloud
            if (tenkan[i] < kijun[i]) or (close[i] < senkou_a[i]) or (close[i] < senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish TK cross or price above cloud
            if (tenkan[i] > kijun[i]) or (close[i] > senkou_a[i]) or (close[i] > senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals
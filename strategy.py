#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Ichimoku Cloud filter with 6h Tenkan-Kijun cross and volume confirmation.
# Goes long when Tenkan crosses above Kijun, price is above 12h Kumo cloud, and volume is above average.
# Goes short when Tenkan crosses below Kijun, price is below 12h Kumo cloud, and volume is above average.
# Uses 12h Ichimoku for trend context and 6s for entry timing to avoid whipsaws.
# Ichimoku works in both bull (trend following with cloud support) and bear (resistance at cloud) markets.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "exp_13799_6h_ichimoku12h_tk_cross_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Tenkan-sen (Conversion Line) period
KIJUN_PERIOD = 26      # Kijun-sen (Base Line) period
SENKOU_SPAN_B_PERIOD = 52  # Senkou Span B period
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward KIJUN_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_SPAN_B_PERIOD shifted forward KIJUN_PERIOD
    senkou_span_b = ((pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

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
    
    # Load 12h data for Ichimoku Cloud (trend filter) ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku Cloud
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tenkan_12h, kijun_12h, senkou_a_12h, senkou_b_12h = calculate_ichimoku(high_12h, low_12h, close_12h)
    
    # Align 12h Ichimoku to 6h timeframe
    tenkan_12h_aligned = align_htf_to_ltf(prices, df_12h, tenkan_12h)
    kijun_12h_aligned = align_htf_to_ltf(prices, df_12h, kijun_12h)
    senkou_a_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_a_12h)
    senkou_b_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_b_12h)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, VOLUME_MA_PERIOD) + KIJUN_PERIOD + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_12h_aligned[i]) or np.isnan(kijun_12h_aligned[i]) or 
            np.isnan(senkou_a_12h_aligned[i]) or np.isnan(senkou_b_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
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
        
        # Ichimoku signals
        # Price above/below Kumo cloud
        cloud_top = np.maximum(senkou_a_12h_aligned[i], senkou_b_12h_aligned[i])
        cloud_bottom = np.minimum(senkou_a_12h_aligned[i], senkou_b_12h_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan-Kijun cross
        tk_cross_above = tenkan_12h_aligned[i] > kijun_12h_aligned[i]
        tk_cross_below = tenkan_12h_aligned[i] < kijun_12h_aligned[i]
        
        # Entry signals
        long_signal = volume_ok and price_above_cloud and tk_cross_above
        short_signal = volume_ok and price_below_cloud and tk_cross_below
        
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
            # Exit long when price falls below cloud or Tenkan crosses below Kijun
            if close[i] < cloud_bottom or tenkan_12h_aligned[i] < kijun_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price rises above cloud or Tenkan crosses above Kijun
            if close[i] > cloud_top or tenkan_12h_aligned[i] > kijun_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals
#!/usr/bin/env python3
"""
exp_6691_6h_ichimoku_cloud_1d_v1
Hypothesis: 6h Ichimoku cloud breakout with 1-day cloud filter and volume confirmation.
Uses 1-day Ichimoku (Tenkan/Kijun/Senkou Span A/B) to determine trend direction.
In 6h timeframe: go long when price breaks above cloud with bullish TK cross and volume,
go short when price breaks below cloud with bearish TK cross and volume.
1-day trend filter ensures alignment with higher timeframe momentum.
Designed for 6h timeframe to capture medium-term swings while minimizing fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6691_6h_ichimoku_cloud_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD_FAST = 9   # Tenkan-sen period
TK_PERIOD_SLOW = 26  # Kijun-sen period
SENKOU_PERIOD = 52   # Senkou span period
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8    # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TK_PERIOD_FAST
    tenkan_sen = (pd.Series(high_1d).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).max() +
                  pd.Series(low_1d).rolling(window=TK_PERIOD_FAST, min_periods=TK_PERIOD_FAST).min()) / 2.0
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over TK_PERIOD_SLOW
    kijun_sen = (pd.Series(high_1d).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).max() +
                 pd.Series(low_1d).rolling(window=TK_PERIOD_SLOW, min_periods=TK_PERIOD_SLOW).min()) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward TK_PERIOD_SLOW periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(TK_PERIOD_SLOW)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_PERIOD shifted forward TK_PERIOD_SLOW
    senkou_span_b = ((pd.Series(high_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() +
                      pd.Series(low_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2.0).shift(TK_PERIOD_SLOW)
    
    # Align HTF Ichimoku to LTF (6h) with shift(1) for completed days only
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TK_PERIOD_SLOW, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + TK_PERIOD_SLOW
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine Ichimoku signals
        # Cloud top/bottom
        cloud_top = np.maximum(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = np.minimum(span_a_aligned[i], span_b_aligned[i])
        
        # TK cross
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) and (close[i] <= cloud_top)
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout signals
        long_breakout = price_above_cloud and tk_bullish and vol_confirmed
        short_breakout = price_below_cloud and tk_bearish and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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
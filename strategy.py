#!/usr/bin/env python3
"""
exp_6759_6h_ichimoku_kijun_sen_1d_trend_v1
Hypothesis: 6h Ichimoku Kijun-Sen (base line) crossover with 1d trend filter.
In bull markets (price > 1d EMA200): long when price crosses above Kijun-Sen.
In bear markets (price < 1d EMA200): short when price crosses below Kijun-Sen.
Uses 6h Tenkan/Kijun cross for timing and 1d EMA200 for structural trend.
Designed to capture medium-term swings in both bull and bear markets with ~12-37 trades/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6759_6h_ichimoku_kijun_sen_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9      # Tenkan-sen period
KJ_PERIOD = 26     # Kijun-sen period
SS_PERIOD = 52     # Senkou Span B period
DISPLACEMENT = 26  # Kumo displacement
EMA_PERIOD = 200   # 1d EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 50  # ~12.5 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (6h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max().values
    period9_low = pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max().values
    period26_low = pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).max().values
    period52_low = pd.Series(low).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
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
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SS_PERIOD, EMA_PERIOD, ATR_PERIOD) + DISPLACEMENT
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
            
        # Determine trend direction from 1d EMA200
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Ichimoku signals: Tenkan/Kijun cross
        # Avoid look-ahead: use previous bar values for cross detection
        tenkan_prev = tenkan_sen[i-1]
        kijun_prev = kijun_sen[i-1]
        tenkan_curr = tenkan_sen[i]
        kijun_curr = kijun_sen[i]
        
        # Bullish cross: Tenkan crosses above Kijun
        bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
        # Bearish cross: Tenkan crosses below Kijun
        bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
        
        # Enter new positions only if flat
        if position == 0:
            if bullish_trend and bullish_cross:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif bearish_trend and bearish_cross:
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
#!/usr/bin/env python3
"""
Experiment #11627: 6h Ichimoku Cloud Breakout with 1d Trend and Volume Confirmation
Hypothesis: Ichimoku cloud provides dynamic support/resistance. Breakouts above/below cloud
with 1d trend filter and volume confirmation capture strong trends. Works in bull (breakouts
continue) and bear (breakouts reverse quickly) by using 1d trend filter. Target: 80-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11627_6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENkou_B = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    tenkan = (pd.Series(high).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).max() +
              pd.Series(low).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    kijun = (pd.Series(high).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).max() +
             pd.Series(low).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods
    senkou_b = (pd.Series(high).rolling(window=ICHIMOKU_SENkou_B, min_periods=ICHIMOKU_SENkou_B).max() +
                pd.Series(low).rolling(window=ICHIMOKU_SENkou_B, min_periods=ICHIMOKU_SENkou_B).min()) / 2
    
    # Current cloud (Senkou Span shifted back to align with price)
    # For cloud at current point, we use Senkou A/B from 26 periods ago
    senkou_a_shifted = senkou_a.shift(ICHIMOKU_KIJUN)
    senkou_b_shifted = senkou_b.shift(ICHIMOKU_KIJUN)
    
    return tenkan.values, kijun.values, senkou_a_shifted.values, senkou_b_shifted.values

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
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ICHIMOKU_SENkou_B, ICHIMOKU_KIJUN, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Ichimoku conditions
        # Price above cloud = bullish, below cloud = bearish
        above_cloud = (close[i] > senkou_a[i] and close[i] > senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        below_cloud = (close[i] < senkou_a[i] and close[i] < senkou_b[i]) if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])) else False
        
        # TK Cross
        tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1]) if i > 0 and not (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1])) else False
        tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1]) if i > 0 and not (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1])) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (1d)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = above_cloud and tk_cross_up and volume_ok and uptrend_1d
        short_entry = below_cloud and tk_cross_down and volume_ok and downtrend_1d
        
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
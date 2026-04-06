#!/usr/bin/env python3
"""
Experiment #11795: 6h Ichimoku Cloud with 1d Kumo Twist and Volume Filter
Hypothesis: Ichimoku system identifies trend via Kumo (cloud) and momentum via TK cross.
1d Kumo twist (Senkou A/B cross) filters for major trend changes, while 6h TK cross
provides timely entries. Volume filter ensures institutional participation.
Works in bull (TK cross above cloud) and bear (TK cross below cloud) via 1d trend filter.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import

name = "exp_11795_6h_ichimoku_kumo_twist_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26      # Kijun-sen (Base Line)
SENKOU_B_PERIOD = 52   # Senkou Span B
KUMO_TWIST_LOOKBACK = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (HH+LL)/2 for past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (HH+LL)/2 for past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (HH+LL)/2 for past 52 periods shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    # Current Kumo (cloud) - Senkou Span A/B from 26 periods ago
    senkou_a_current = senkou_a.shift(-KIJUN_PERIOD)
    senkou_b_current = senkou_b.shift(-KIJUN_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a_current.values, senkou_b_current.values

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter (using close)
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo twist detection on 1d: Senkou A/B cross
    # We need to detect when Senkou A crosses Senkou B on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku on 1d for Kumo twist
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(KIJUN_PERIOD)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                    pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    # Current Kumo on 1d (shifted back)
    senkou_a_1d_current = senkou_a_1d.shift(-KIJUN_PERIOD)
    senkou_b_1d_current = senkou_b_1d.shift(-KIJUN_PERIOD)
    
    # Kumo twist: Senkou A crosses Senkou B
    # Twist up: Senkou A crosses above Senkou B (bullish)
    # Twist down: Senkou A crosses below Senkou B (bearish)
    kumo_twist_up = np.zeros(len(df_1d), dtype=bool)
    kumo_twist_down = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(1, len(df_1d)):
        if not np.isnan(senkou_a_1d_current[i]) and not np.isnan(senkou_b_1d_current[i]):
            if not np.isnan(senkou_a_1d_current[i-1]) and not np.isnan(senkou_b_1d_current[i-1]):
                kumo_twist_up[i] = (senkou_a_1d_current[i-1] <= senkou_b_1d_current[i-1] and 
                                   senkou_a_1d_current[i] > senkou_b_1d_current[i])
                kumo_twist_down[i] = (senkou_a_1d_current[i-1] >= senkou_b_1d_current[i-1] and 
                                     senkou_a_1d_current[i] < senkou_b_1d_current[i])
    
    # Align Kumo twist to 6h
    kumo_twist_up_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_up.astype(float))
    kumo_twist_down_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_down.astype(float))
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD) + KIJUN_PERIOD + 1
    
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
        
        # Ichimoku signals
        # TK Cross: Tenkan crosses Kijun
        tk_cross_up = (tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]) if i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]) else False
        tk_cross_down = (tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]) if i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]) else False
        
        # Price relative to Kumo
        price_above_kumo = not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]) and close[i] > max(senkou_a[i], senkou_b[i])
        price_below_kumo = not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]) and close[i] < min(senkou_a[i], senkou_b[i])
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (1d EMA)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Kumo twist filter (1d)
        twist_up = kumo_twist_up_aligned[i] > 0.5 if not np.isnan(kumo_twist_up_aligned[i]) else False
        twist_down = kumo_twist_down_aligned[i] > 0.5 if not np.isnan(kumo_twist_down_aligned[i]) else False
        
        # Entry conditions
        # Long: TK cross up + price above Kumo + volume + uptrend 1d + no recent twist down
        long_entry = (tk_cross_up and price_above_kumo and volume_ok and uptrend_1d and 
                     not twist_down)
        
        # Short: TK cross down + price below Kumo + volume + downtrend 1d + no recent twist up
        short_entry = (tk_cross_down and price_below_kumo and volume_ok and downtrend_1d and 
                      not twist_up)
        
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
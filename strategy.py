#!/usr/bin/env python3
"""
Experiment #11151: 6h Ichimoku Cloud Breakout with 1d Kumo Filter
Hypothesis: Ichimoku captures momentum and support/resistance. Daily Kumo (cloud) provides trend bias.
Breakouts above/below Kumo with TK cross in same direction capture strong moves. Works in bull (cloud acts as support)
and bear (cloud acts as resistance) by using 1d cloud filter. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11151_6h_ichimoku_kumo_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(-KUMO_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Kumo (cloud) for trend filter
    # We need daily high, low for Senkou Span A/B
    d_high = df_daily['high'].values
    d_low = df_daily['low'].values
    d_close = df_daily['close'].values
    
    # Daily Ichimoku components (using same periods)
    d_tenkan = (pd.Series(d_high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                pd.Series(d_low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    d_kijun = (pd.Series(d_high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
               pd.Series(d_low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    d_senkou_a = ((d_tenkan + d_kijun) / 2).shift(KUMO_SHIFT)
    d_senkou_b = ((pd.Series(d_high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                   pd.Series(d_low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Daily Kumo edges (cloud top and bottom)
    d_kumo_top = np.maximum(d_senkou_a.values, d_senkou_b.values)
    d_kumo_bottom = np.minimum(d_senkou_a.values, d_senkou_b.values)
    
    # Align daily Kumo to 6h timeframe
    d_kumo_top_aligned = align_htf_to_ltf(prices, df_daily, d_kumo_top)
    d_kumo_bottom_aligned = align_htf_to_ltf(prices, df_daily, d_kumo_bottom)
    
    # Calculate 6h Ichimoku
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # 6h Kumo edges
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))  # Tenkan crosses above Kijun
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))  # Tenkan crosses below Kijun
    
    # Price relative to Kumo
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Daily trend filter: price relative to daily Kumo
    price_above_daily_kumo = close > d_kumo_top_aligned
    price_below_daily_kumo = close < d_kumo_bottom_aligned
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) * 2  # Extra buffer for Ichimoku
    
    for i in range(start, n):
        # Skip if daily Kumo not available
        if np.isnan(d_kumo_top_aligned[i]) or np.isnan(d_kumo_bottom_aligned[i]):
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
        
        # Entry conditions
        # Long: price above both Kumo AND TK cross up OR strong bullish alignment
        long_condition = (price_above_kumo[i] and price_above_daily_kumo[i] and 
                         (tk_cross_up[i] or (tenkan[i] > kijun[i] and tenkan[i-1] > kijun[i-1])))
        
        # Short: price below both Kumo AND TK cross down OR strong bearish alignment
        short_condition = (price_below_kumo[i] and price_below_daily_kumo[i] and 
                          (tk_cross_down[i] or (tenkan[i] < kijun[i] and tenkan[i-1] < kijun[i-1])))
        
        # Generate signals
        if position == 0:
            if long_condition:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_condition:
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
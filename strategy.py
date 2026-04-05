#!/usr/bin/env python3
"""
Experiment #7891: 6-hour Ichimoku Cloud with Daily Trend Filter
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction. 
Tenkan/Kijun cross signals momentum shifts. Daily trend filter (price vs Kumo) 
avoids counter-trend trades. Works in both bull/bear markets by adapting to 
cloud thickness and price-position relative to cloud. Targets 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7891_6h_ichimoku1d_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Conversion Line
KIJUN_PERIOD = 26    # Base Line
SENKOU_B_PERIOD = 52 # Leading Span B
KUMO_SHIFT = 26      # Cloud displacement
SIGNAL_SIZE = 0.25   # Position size (25% of capital)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - Daily for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan_sen = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun_sen = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted KUMO_SHIFT periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_B_PERIOD shifted KUMO_SHIFT
    senkou_b = ((pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Current Kumo (cloud) top and bottom - shifted back to align with current price
    kumo_top = np.maximum(senkou_a.values, senkou_b.values)
    kumo_bottom = np.minimum(senkou_a.values, senkou_b.values)
    
    # Price relative to Kumo: above=bullish, below=bearish, inside=neutral
    price_above_kumo = close_1d > kumo_top
    price_below_kumo = close_1d < kumo_bottom
    
    # Align daily Ichimoku signals to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    price_above_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_above_kumo.astype(float))
    price_below_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_below_kumo.astype(float))
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Tenkan and Kijun on 6h for entry signals
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + KUMO_SHIFT + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Get current values
        tenkan_t = tenkan_6h[i]
        kijun_t = kijun_6h[i]
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        price_above = price_above_kumo_aligned[i] > 0.5
        price_below = price_below_kumo_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = tk_up and price_above
        short_entry = tk_down and price_below
        
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
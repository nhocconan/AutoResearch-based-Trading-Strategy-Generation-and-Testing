#!/usr/bin/env python3
"""
Experiment #8355: 6-hour Ichimoku cloud with 1-week trend filter and volume confirmation.
Hypothesis: Price breaking above/below the Ichimoku cloud on 6h with volume >1.5x 20-period MA 
and aligned 1-week trend (price above/below weekly Kumo) captures sustained moves while 
avoiding whipsaw. The weekly trend filter provides long-term context, reducing false 
breakouts during consolidation. Targeting 75-200 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8355_6h_ichimoku_1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Conversion Line
KIJUN_PERIOD = 26      # Base Line
SENKOU_B_PERIOD = 52   # Leading Span B
SENKOU_SHIFT = 26      # Leading Span shift
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(SENKOU_SHIFT)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_1w).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                      pd.Series(low_1w).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(SENKOU_SHIFT)
    
    # Kumo (Cloud) top and bottom
    kumO_top = np.maximum(senkou_span_a.values, senkou_span_b.values)
    kumO_bottom = np.minimum(senkou_span_a.values, senkou_span_b.values)
    
    # Price relative to Kumo: above = bullish bias, below = bearish bias
    price_vs_kumo = np.where(close_1w > kumO_top, 1, 
                     np.where(close_1w < kumO_bottom, -1, 0))  # 1=bullish, -1=bearish, 0=in cloud
    price_vs_kumo_aligned = align_htf_to_ltf(prices, df_1w, price_vs_kumo)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h
    # Tenkan-sen (Conversion Line)
    tenkan_sen_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                     pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line)
    kijun_sen_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                    pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A)
    senkou_span_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2).shift(SENKOU_SHIFT)
    # Senkou Span B (Leading Span B)
    senkou_span_b_6h = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                         pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(SENKOU_SHIFT)
    
    # Kumo (Cloud) top and bottom
    kumO_top_6h = np.maximum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    kumO_bottom_6h = np.minimum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD + SENKOU_SHIFT, 
                VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_kumo_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1w Kumo
        bull_bias = price_vs_kumo_aligned[i] == 1   # 1w price above Kumo
        bear_bias = price_vs_kumo_aligned[i] == -1  # 1w price below Kumo
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond cloud bands
        cloud_breakout_up = (close[i] > kumO_top_6h[i-1]) and (i-1 >= 0) and not np.isnan(kumO_top_6h[i-1])
        cloud_breakout_down = (close[i] < kumO_bottom_6h[i-1]) and (i-1 >= 0) and not np.isnan(kumO_bottom_6h[i-1])
        
        # Entry conditions
        long_entry = bull_bias and cloud_breakout_up and volume_confirmed
        short_entry = bear_bias and cloud_breakout_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
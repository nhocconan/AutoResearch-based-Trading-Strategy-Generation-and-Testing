#!/usr/bin/env python3
"""
Experiment #8295: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation.
Hypothesis: In crypto markets, Ichimoku Cloud provides robust support/resistance levels 
that work in both bull and bear markets. Price above/below the weekly cloud determines 
trend direction, while Tenkan-Kijun cross on 6h provides entry timing with volume 
confirmation to avoid false signals. Weekly trend filter reduces whipsaw during 
consolidation. Targets 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8295_6h_ichimoku1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Determine trend: price above/both Senkou spans = bullish, below/both = bearish
    # Price above cloud = bullish, below cloud = bearish
    cloud_top = np.maximum(senkou_a_1w, senkou_b_1w)
    cloud_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    price_above_cloud = close_1w > cloud_top
    price_below_cloud = close_1w < cloud_bottom
    
    # Trend: 1 = bullish (price above cloud), -1 = bearish (price below cloud), 0 = in cloud (no trend)
    weekly_trend = np.where(price_above_cloud, 1, np.where(price_below_cloud, -1, 0))
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Previous bar condition handled separately
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Previous bar condition handled separately
    
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
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_trend_aligned[i]):
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
        
        # Determine weekly trend bias
        bull_bias = weekly_trend_aligned[i] == 1   # Weekly price above cloud
        bear_bias = weekly_trend_aligned[i] == -1  # Weekly price below cloud
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # TK Cross conditions (check actual crossover)
        tk_cross_up_actual = False
        tk_cross_down_actual = False
        if i > 0:
            tk_cross_up_actual = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
            tk_cross_down_actual = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
        
        # Entry conditions
        long_entry = bull_bias and tk_cross_up_actual and volume_confirmed
        short_entry = bear_bias and tk_cross_down_actual and volume_confirmed
        
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
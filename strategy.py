#!/usr/bin/env python3
"""
Experiment #8999: 6h Donchian(20) breakout + 12h Ichimoku cloud filter + volume confirmation
Hypothesis: Donchian breakouts capture trends; 12h Ichimoku cloud filters trend direction (bullish when price above cloud, bearish when below); volume confirms institutional participation.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost. Works in bull (breakouts above cloud) and bear (breakouts below cloud).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8999_6h_donchian20_12h_ichimoku_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ICHIMOKU_CONVERSION = 9
ICHIMOKU_BASE = 26
ICHIMOKU_LEADING_SPAN_B = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=ICHIMOKU_CONVERSION, min_periods=ICHIMOKU_CONVERSION).max()
    low_9 = pd.Series(low).rolling(window=ICHIMOKU_CONVERSION, min_periods=ICHIMOKU_CONVERSION).min()
    conversion_line = (high_9 + low_9) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=ICHIMOKU_BASE, min_periods=ICHIMOKU_BASE).max()
    low_26 = pd.Series(low).rolling(window=ICHIMOKU_BASE, min_periods=ICHIMOKU_BASE).min()
    base_line = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2
    leading_span_a = (conversion_line + base_line) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(high).rolling(window=ICHIMOKU_LEADING_SPAN_B, min_periods=ICHIMOKU_LEADING_SPAN_B).max()
    low_52 = pd.Series(low).rolling(window=ICHIMOKU_LEADING_SPAN_B, min_periods=ICHIMOKU_LEADING_SPAN_B).min()
    leading_span_b = (high_52 + low_52) / 2
    
    return conversion_line.values, base_line.values, leading_span_a.values, leading_span_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku Cloud
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    conversion_line, base_line, leading_span_a, leading_span_b = calculate_ichimoku(high_12h, low_12h, close_12h)
    
    # Cloud top and bottom (Leading Span A and B)
    cloud_top = np.maximum(leading_span_a, leading_span_b)
    cloud_bottom = np.minimum(leading_span_a, leading_span_b)
    
    # Price relative to cloud: above = bullish, below = bearish, inside = neutral
    price_vs_cloud = np.where(close_12h > cloud_top, 1, 
                     np.where(close_12h < cloud_bottom, -1, 0))  # 1=bullish, -1=bearish, 0=inside cloud
    price_vs_cloud_aligned = align_htf_to_ltf(prices, df_12h, price_vs_cloud)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ICHIMOKU_LEADING_SPAN_B, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_cloud_aligned[i]):
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
        
        # Determine market bias from 12h Ichimoku cloud
        bull_bias = price_vs_cloud_aligned[i] == 1   # 12h price above cloud
        bear_bias = price_vs_cloud_aligned[i] == -1  # 12h price below cloud
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
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
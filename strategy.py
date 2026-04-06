#!/usr/bin/env python3
"""
Experiment #11975: 6h Ichimoku Cloud with 1w Trend Filter
Hypothesis: Ichimoku Cloud provides dynamic support/resistance and trend direction.
Using 1w Ichimoku as primary trend filter reduces false breakouts in sideways markets.
6s line (Tenkan-sen) crossing above/below base line (Kijun-sen) within the cloud
provides momentum signals with trend alignment. Works in bull (cloud acts as support) 
and bear (cloud acts as resistance) by using weekly cloud color as regime filter.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11975_6h_ichimoku_1w_trend"
timeframe = "6h"
leverage = 1.0

# Ichimoku parameters
TENKAN_PERIOD = 9      # Conversion Line
KIJUN_PERIOD = 26      # Base Line
SENKOU_B_PERIOD = 52   # Leading Span B
KUMO_SHIFT = 26        # Cloud displacement

# Signal parameters
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
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
    
    # Chikou Span (Lagging Span): Close shifted -26 periods
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku for trend filter
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w, chikou_w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    
    # Determine cloud color (green = bullish, red = bearish)
    # Bullish when Senkou A > Senkou B
    cloud_bullish = senkou_a_w > senkou_b_w
    cloud_bearish = senkou_a_w < senkou_b_w
    
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1w, cloud_bullish.astype(float))
    cloud_bearish_aligned = align_htf_to_ltf(prices, df_1w, cloud_bearish.astype(float))
    
    # Calculate 6h Ichimoku for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Calculate ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for Ichomoku calculations
    start = max(SENKOU_B_PERIOD + KUMO_SHIFT, KIJUN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1w Ichimoku not available
        if np.isnan(cloud_bullish_aligned[i]) or np.isnan(cloud_bearish_aligned[i]):
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
        
        # Skip if Ichimoku lines not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signals
        # Tenkan-sen crossing above/below Kijun-sen
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] if i > 0 else False
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] if i > 0 else False
        
        # Price relative to cloud
        price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        price_in_cloud = (close[i] >= min(senkou_a[i], senkou_b[i]) and 
                         close[i] <= max(senkou_a[i], senkou_b[i]))
        
        # Entry conditions with weekly trend filter
        # Long: TK cross up + price above/below cloud + weekly bullish cloud
        long_entry = (tk_cross_up and 
                     (price_above_cloud or price_in_cloud) and 
                     cloud_bullish_aligned[i] > 0.5)
        
        # Short: TK cross down + price above/below cloud + weekly bearish cloud
        short_entry = (tk_cross_down and 
                      (price_below_cloud or price_in_cloud) and 
                      cloud_bearish_aligned[i] > 0.5)
        
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
            # Stay in long position
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Stay in short position
            signals[i] = -SIGNAL_SIZE
    
    return signals
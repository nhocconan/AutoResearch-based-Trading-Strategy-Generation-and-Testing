#!/usr/bin/env python3
"""
Experiment #8111: 6-hour Ichimoku Cloud with 1-day filter and volume confirmation.
Hypothesis: Price crossing above/below Tenkan-sen (conversion line) on 6h with price outside Kumo (cloud) from 1d and volume >1.5x 20-period MA captures trend continuation with controlled frequency. Uses 1d Ichimoku for stronger trend filter than 12h, reducing whipsaw while targeting 75-200 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8111_6h_ichimoku1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Conversion line
KIJUN_PERIOD = 26      # Base line
SENKOU_B_PERIOD = 52   # Leading span B
KUMO_SHIFT = 26        # Cloud shift
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                     pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_span_a_shifted = senkou_span_a.shift(KUMO_SHIFT)
    senkou_span_b_shifted = senkou_span_b.shift(KUMO_SHIFT)
    
    # Determine Kumo (cloud) boundaries
    kumo_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    kumo_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # Price above/below cloud: 1=above (bullish), -1=below (bearish), 0=inside (neutral)
    price_vs_kumo = np.where(close_1d > kumo_top, 1, np.where(close_1d < kumo_bottom, -1, 0))
    price_vs_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_vs_kumo)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Tenkan-sen on 6h
    tenkan_sen_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                     pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
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
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Determine market bias from 1d Ichimoku cloud
        bull_bias = price_vs_kumo_aligned[i] == 1   # 1d price above cloud
        bear_bias = price_vs_kumo_aligned[i] == -1  # 1d price below cloud
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Tenkan-sen cross conditions - require close cross to avoid wicks
        tenkan_cross_up = (close[i] > tenkan_sen_6h[i]) and (close[i-1] <= tenkan_sen_6h[i-1]) if i-1 >= 0 and not np.isnan(tenkan_sen_6h[i]) and not np.isnan(tenkan_sen_6h[i-1]) else False
        tenkan_cross_down = (close[i] < tenkan_sen_6h[i]) and (close[i-1] >= tenkan_sen_6h[i-1]) if i-1 >= 0 and not np.isnan(tenkan_sen_6h[i]) and not np.isnan(tenkan_sen_6h[i-1]) else False
        
        # Entry conditions
        long_entry = bull_bias and tenkan_cross_up and volume_confirmed
        short_entry = bear_bias and tenkan_cross_down and volume_confirmed
        
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
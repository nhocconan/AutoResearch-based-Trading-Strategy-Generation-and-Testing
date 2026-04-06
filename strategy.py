#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour trading based on weekly Ichimoku Cloud with daily price action confirmation.
# The weekly Ichimoku provides strong trend context (bullish when price above cloud, bearish when below).
# Daily price action (close relative to weekly Tenkan/Kijun) filters for high-probability entries.
# This combination reduces false signals in choppy markets while capturing strong trends in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "ichimoku_6h_weekly_daily_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Tenkan-sen (Conversion Line) period
KIJUN_PERIOD = 26    # Kijun-sen (Base Line) period
SENKOU_B_PERIOD = 52 # Senkou Span B period
CHIKOU_PERIOD = 26   # Chikou Span lag
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward KIJUN_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KIJUN_PERIOD)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_B_PERIOD shifted forward KIJUN_PERIOD
    senkou_span_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    
    # Chikou Span (Lagging Span): Close shifted backward CHIKOU_PERIOD
    chikou_span = pd.Series(close).shift(-CHIKOU_PERIOD)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    chikou_1w_aligned = align_htf_to_ltf(prices, df_1w, chikou_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period - need enough data for Ichimoku
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KIJUN_PERIOD*2, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly Ichimoku data not available
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # Ichimoku signals:
        # Bullish: price above cloud AND Tenkan > Kijun
        # Bearish: price below cloud AND Tenkan < Kijun
        bullish_setup = (close[i] > cloud_top) and (tenkan_1w_aligned[i] > kijun_1w_aligned[i])
        bearish_setup = (close[i] < cloud_bottom) and (tenkan_1w_aligned[i] < kijun_1w_aligned[i])
        
        # Additional filter: Chikou Span confirmation (price was above/below CHIKOU_PERIOD ago)
        chikou_bullish = close[i - CHIKOU_PERIOD] > chikou_1w_aligned[i - CHIKOU_PERIOD] if i >= CHIKOU_PERIOD else False
        chikou_bearish = close[i - CHIKOU_PERIOD] < chikou_1w_aligned[i - CHIKOU_PERIOD] if i >= CHIKOU_PERIOD else False
        
        # Final entry conditions
        bullish_entry = bullish_setup and volume_ok and chikou_bullish
        bearish_entry = bearish_setup and volume_ok and chikou_bearish
        
        # Generate signals
        if position == 0:
            if bullish_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_entry:
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
#!/usr/bin/env python3
"""
Experiment #10335: 6h Ichimoku Cloud + Weekly Trend + Volume Spike
Hypothesis: Ichimoku Tenkan/Kijun cross in the direction of weekly trend (above/below cloud)
with volume confirmation provides high-probability trend continuation trades.
Works in bull markets (bullish cross above cloud) and bear markets (bearish cross below cloud).
Volume filters reduce false signals. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10335_6h_ichimoku_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_SPIKE_MULTIPLIER = 1.8
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

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
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku for trend direction
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_tenkan, weekly_kijun, weekly_senkou_a, weekly_senkou_b = calculate_ichimoku(weekly_high, weekly_low, weekly_close)
    
    # Align weekly Ichimoku to 6h timeframe
    weekly_tenkan_aligned = align_htf_to_ltf(prices, df_weekly, weekly_tenkan)
    weekly_kijun_aligned = align_htf_to_ltf(prices, df_weekly, weekly_kijun)
    weekly_senkou_a_aligned = align_htf_to_ltf(prices, df_weekly, weekly_senkou_a)
    weekly_senkou_b_aligned = align_htf_to_ltf(prices, df_weekly, weekly_senkou_b)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Ichimoku
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD, KIJUN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly Ichimoku not available
        if (np.isnan(weekly_tenkan_aligned[i]) or np.isnan(weekly_kijun_aligned[i]) or
            np.isnan(weekly_senkou_a_aligned[i]) or np.isnan(weekly_senkou_b_aligned[i])):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter: price above/both Senkou spans (cloud)
        weekly_cloud_top = np.maximum(weekly_senkou_a_aligned[i], weekly_senkou_b_aligned[i])
        weekly_cloud_bottom = np.minimum(weekly_senkou_a_aligned[i], weekly_senkou_b_aligned[i])
        price_above_weekly_cloud = close[i] > weekly_cloud_top
        price_below_weekly_cloud = close[i] < weekly_cloud_bottom
        
        # Ichimoku signals: Tenkan/Kijun cross
        bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Entry conditions: Ichimoku cloud + cross in direction of weekly trend
        long_entry = bullish_cross and price_above_weekly_cloud and volume_spike
        short_entry = bearish_cross and price_below_weekly_cloud and volume_spike
        
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
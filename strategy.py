#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12575_6d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Conversion line
KIJUN_PERIOD = 26    # Base line
SENKOU_B_PERIOD = 52 # Span B
KUMO_SHIFT = 26      # Kumo shift
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_ichimoku(high, low):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                     pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w = calculate_ichimoku(df_1w['high'].values, df_1w['low'].values)
    
    # Align to 6h timeframe
    tenkan_w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_aligned = align_htf_to_ltf(prices, df_1w, kijun_w)
    senkou_a_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_w)
    senkou_b_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT, 
                VOLUME_MA_PERIOD, ATR_PERIOD) + KUMO_SHIFT + 1
    
    for i in range(start, n):
        # Skip if weekly Ichimoku not available
        if (np.isnan(tenkan_w_aligned[i]) or np.isnan(kijun_w_aligned[i]) or 
            np.isnan(senkou_a_w_aligned[i]) or np.isnan(senkou_b_w_aligned[i])):
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
        
        # Weekly trend filter (Ichimoku cloud)
        # Bullish: price above cloud, Tenkan > Kijun
        # Bearish: price below cloud, Tenkan < Kijun
        price_above_cloud = close[i] > max(senkou_a_w_aligned[i], senkou_b_w_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_w_aligned[i], senkou_b_w_aligned[i])
        bullish_cross = tenkan_w_aligned[i] > kijun_w_aligned[i]
        bearish_cross = tenkan_w_aligned[i] < kijun_w_aligned[i]
        
        weekly_uptrend = price_above_cloud and bullish_cross
        weekly_downtrend = price_below_cloud and bearish_cross
        
        # Entry conditions
        long_entry = volume_ok and weekly_uptrend
        short_entry = volume_ok and weekly_downtrend
        
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
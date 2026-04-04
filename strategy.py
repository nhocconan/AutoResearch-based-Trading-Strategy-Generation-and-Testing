#!/usr/bin/env python3
"""
exp_6699_6h_ichimoku_cloud_1d_v1
Hypothesis: 6h Ichimoku cloud breakout with 1d cloud filter and volume confirmation.
Uses 1-day Ichimoku cloud (senkou span A/B) to determine trend direction and cloud
twist (TK cross) on 6h for entry timing. In bullish 1d cloud (price above cloud),
look for 6h TK cross up with volume confirmation for longs. In bearish 1d cloud
(price below cloud), look for 6h TK cross down with volume confirmation for shorts.
Avoids counter-trend trades and whipsaws in ranging markets. Designed for 6h
timeframe with ~25-40 trades/year to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6699_6h_ichimoku_cloud_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9      # Tenkan-sen (Conversion Line) period
KJ_PERIOD = 26     # Kijun-sen (Base Line) period
SENKOU_PERIOD = 52 # Senkou span B period
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 3  # ~18 hours (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TK_PERIOD
    tenkan_sen = (
        pd.Series(high_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max().values +
        pd.Series(low_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min().values
    ) / 2.0
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KJ_PERIOD
    kijun_sen = (
        pd.Series(high_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max().values +
        pd.Series(low_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min().values
    ) / 2.0
    
    # Senkou span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward KJ_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    senkou_span_a = np.roll(senkou_span_a, -KJ_PERIOD)  # shift forward
    
    # Senkou span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_PERIOD shifted forward KJ_PERIOD
    senkou_span_b = (
        pd.Series(high_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max().values +
        pd.Series(low_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min().values
    ) / 2.0
    senkou_span_b = np.roll(senkou_span_b, -KJ_PERIOD)  # shift forward
    
    # Align HTF Ichimoku to LTF (6h) with shift(1) for completed days only
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate LTF indicators (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # LTF Tenkan-sen and Kijun-sen for TK cross
    ltf_tenkan = (
        pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max().values +
        pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min().values
    ) / 2.0
    
    ltf_kijun = (
        pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max().values +
        pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min().values
    ) / 2.0
    
    # TK cross signals: Tenkan crossing Kijun
    tk_cross_up = (ltf_tenkan > ltf_kijun) & (np.roll(ltf_tenkan, 1) <= np.roll(ltf_kijun, 1))
    tk_cross_down = (ltf_tenkan < ltf_kijun) & (np.roll(ltf_tenkan, 1) >= np.roll(ltf_kijun, 1))
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + KJ_PERIOD + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine 1d Ichimoku trend: price relative to cloud
        # Cloud top = max(senkou_span_a, senkou_span_b)
        # Cloud bottom = min(senkou_span_a, senkou_span_b)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        bullish_cloud = close[i] > cloud_top
        bearish_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Entry signals: TK cross in direction of 1d cloud with volume
        long_signal = bullish_cloud and tk_cross_up[i] and vol_confirmed
        short_signal = bearish_cloud and tk_cross_down[i] and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals
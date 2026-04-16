#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d TK Cross and Volume Confirmation
# Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK Cross.
# Using 1d timeframe for Ichimoku calculation reduces noise and provides stronger trend context.
# TK Cross (Tenkan/Kijun cross) on 1d acts as momentum filter for 6h entries.
# Volume confirmation (>1.3x 20-period average) ensures breakouts have participation.
# This combination adapts to both trending and ranging markets with controlled trade frequency.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Ichimoku calculation (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Ichimoku Components (1d) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # === 6h Volume Confirmation (20-period average) ===
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or
            np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        span_a = span_a_aligned[i]
        span_b = span_b_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.3  # 1.3x average volume
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # === TRAILING STOP LOGIC (ATR-based) ===
        # Calculate ATR from 6h data for trailing stop
        if i >= warmup:  # Ensure we have enough data for ATR
            # Simplified ATR calculation using recent 6h data
            atr_period = 14
            if i >= atr_period:
                tr_list = []
                for j in range(max(warmup, i-atr_period+1), i+1):
                    if j < len(prices):
                        tr1 = prices['high'].iloc[j] - prices['low'].iloc[j]
                        tr2 = abs(prices['high'].iloc[j] - prices['close'].iloc[j-1]) if j > 0 else tr1
                        tr3 = abs(prices['low'].iloc[j] - prices['close'].iloc[j-1]) if j > 0 else tr1
                        tr_list.append(max(tr1, tr2, tr3))
                if tr_list:
                    atr_val = np.mean(tr_list)
                else:
                    atr_val = 0.0
            else:
                atr_val = 0.0
        else:
            atr_val = 0.0
        
        # Track extreme price for trailing stop
        if not hasattr(generate_signals, 'extreme_price'):
            generate_signals.extreme_price = 0.0
        
        if position == 1:  # Long position
            # Update extreme price (highest since entry)
            if price > generate_signals.extreme_price:
                generate_signals.extreme_price = price
            # Trail stop: exit if price drops 2.0*ATR from extreme
            if atr_val > 0 and price < generate_signals.extreme_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                generate_signals.extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update extreme price (lowest since entry)
            if price < generate_signals.extreme_price or generate_signals.extreme_price == 0:
                generate_signals.extreme_price = price
            # Trail stop: exit if price rises 2.0*ATR from extreme
            if atr_val > 0 and price > generate_signals.extreme_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                generate_signals.extreme_price = 0.0
                continue
        
        # === EXIT LOGIC (TK Cross reversal) ===
        if position == 1:  # Long position
            # Exit when Tenkan crosses below Kijun (bearish TK cross)
            if tenkan < kijun:
                signals[i] = 0.0
                position = 0
                generate_signals.extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Tenkan crosses above Kijun (bullish TK cross)
            if tenkan > kijun:
                signals[i] = 0.0
                position = 0
                generate_signals.extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish TK Cross: Tenkan crosses above Kijun
            bullish_tk = tenkan > kijun and tenkan_aligned[i-1] <= kijun_aligned[i-1] if i > 0 else False
            # Bearish TK Cross: Tenkan crosses below Kijun
            bearish_tk = tenkan < kijun and tenkan_aligned[i-1] >= kijun_aligned[i-1] if i > 0 else False
            
            # Long when: Bullish TK Cross AND price above cloud AND volume confirmation
            if bullish_tk and price > cloud_top and vol_confirm:
                signals[i] = 0.25
                position = 1
                generate_signals.extreme_price = price
                continue
            # Short when: Bearish TK Cross AND price below cloud AND volume confirmation
            elif bearish_tk and price < cloud_bottom and vol_confirm:
                signals[i] = -0.25
                position = -1
                generate_signals.extreme_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    # Reset class attribute for next call
    if hasattr(generate_signals, 'extreme_price'):
        delattr(generate_signals, 'extreme_price')
    
    return signals

name = "6h_Ichimoku_TKCross_1d_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 1d as trend filter, combined with 6h price breaking above/below cloud and volume confirmation. Works in both bull/bear markets because cloud twist captures major trend changes, while volume confirmation avoids false breakouts. Designed for low trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === Ichimoku components on 1d ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Cloud twist: Senkou Span A crossing above/below Senkou Span B
    # Bullish twist: Senkou A > Senkou B (after previously being below)
    # Bearish twist: Senkou A < Senkou B (after previously being above)
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = senkou_a[0]
    senkou_b_prev[0] = senkou_b[0]
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Current cloud top/bottom for price position
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for dynamic stoploss (14-period on 6h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        is_bullish_twist = bullish_twist_aligned[i] > 0.5
        is_bearish_twist = bearish_twist_aligned[i] > 0.5
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: bullish twist + price above cloud + volume spike > 1.8
            if is_bullish_twist and price_close > cloud_top_val and vol_spike > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: bearish twist + price below cloud + volume spike > 1.8
            elif is_bearish_twist and price_close < cloud_bottom_val and vol_spike > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
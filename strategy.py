#!/usr/bin/env python3
"""
Experiment #031: 6h Ichimoku Cloud + 1d Weekly Pivot Direction Filter
HYPOTHESIS: Ichimoku Cloud (TK cross + price relative to cloud) provides high-probability momentum signals,
while 1d weekly pivot acts as a regime filter - only take longs when price above weekly pivot (bull bias),
only shorts when price below weekly pivot (bear bias). This combines trend momentum with structural bias
to work in both bull and bear markets by avoiding counter-trend trades. Minimum 6-bar holding period
reduces churn. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_031_6h_ichimoku_1d_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    df_1d_idx = pd.RangeIndex(len(df_1d))
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1)   # Prior week low
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(1) # Prior week close
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Current Ichimoku values (no look-ahead - using already shifted components)
    tenkan_sen_vals = tenkan_sen.values
    kijun_sen_vals = kijun_sen.values
    senkou_span_a_vals = senkou_span_a.values
    senkou_span_b_vals = senkou_span_b.values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 60  # Warmup for Ichimoku stability (need 52+26 periods)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(tenkan_sen_vals[i]) or np.isnan(kijun_sen_vals[i]) or
            np.isnan(senkou_span_a_vals[i]) or np.isnan(senkou_span_b_vals[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Ichimoku Components ---
        tenkan = tenkan_sen_vals[i]
        kijun = kijun_sen_vals[i]
        span_a = senkou_span_a_vals[i]
        span_b = senkou_span_b_vals[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK Cross (Tenkan-sen crossing Kijun-sen)
        tk_cross_up = tenkan > kijun and tenkan_sen_vals[i-1] <= kijun_sen_vals[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_vals[i-1] >= kijun_sen_vals[i-1]
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        price_in_cloud = (price >= cloud_bottom) and (price <= cloud_top)
        
        # --- Weekly Pivot Bias: Only trade when price is clearly above/below weekly pivot ---
        is_above_pivot = price > weekly_pivot_aligned[i] * 1.001  # 0.1% buffer above pivot
        is_below_pivot = price < weekly_pivot_aligned[i] * 0.999  # 0.1% buffer below pivot
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on TK cross down (contrarian exit)
                if tk_cross_down:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on TK cross up (contrarian exit)
                if tk_cross_up:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 6 bars to reduce churn
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: TK cross up + price above cloud + price above weekly pivot
        if tk_cross_up and price_above_cloud and is_above_pivot:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: TK cross down + price below cloud + price below weekly pivot
        elif tk_cross_down and price_below_cloud and is_below_pivot:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
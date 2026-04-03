#!/usr/bin/env python3
"""
Experiment #067: 6h Ichimoku Cloud + 1d ADX Trend Filter + Volume Spike

HYPOTHESIS: Ichimoku system (Tenkan/Kijun cross + price vs cloud) on 6h timeframe, 
filtered by 1d ADX > 25 for trending markets and 12h volume spike > 1.8x average, 
creates a high-probability trend-following strategy. Ichimoku provides dynamic 
support/resistance, ADX ensures we only trade when trends are strong (reducing 
whipsaw in ranging markets), and volume confirms institutional participation. 
Targets 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize 
fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx = np.where((di_plus + di_minus) == 0, 0, dx)
        adx_14 = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    else:
        adx_14_aligned = np.full(n, 0.0)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if len(high) >= period_tenkan and len(low) >= period_tenkan:
        tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                     pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
        tenkan_sen = tenkan_sen.values
    else:
        tenkan_sen = np.full(n, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if len(high) >= period_kijun and len(low) >= period_kijun:
        kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                    pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
        kijun_sen = kijun_sen.values
    else:
        kijun_sen = np.full(n, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    if len(high) >= period_senkou_b and len(low) >= period_senkou_b:
        senkou_span_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                        pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
        senkou_span_b = senkou_span_b.values
    else:
        senkou_span_b = np.full(n, np.nan)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = np.roll(close, -26)  # Will handle alignment in signals
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Ichimoku Components ---
        # Cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) & (close[i] <= cloud_top)
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan_sen[i] > kijun_sen[i]
        tenkan_below_kijun = tenkan_sen[i] < kijun_sen[i]
        
        # --- Regime Filter: Only trade when ADX > 25 (strong trend) ---
        strong_trend = adx_14_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # --- Exit Logic (Ichimoku-based exit) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when price falls below cloud OR Tenkan crosses below Kijun
                if price_below_cloud or (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when price rises above cloud OR Tenkan crosses above Kijun
                if price_above_cloud or (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price above cloud + Tenkan above Kijun + strong trend + volume spike
        long_condition = (
            price_above_cloud and 
            tenkan_above_kijun and 
            strong_trend and 
            volume_spike
        )
        
        # Short: Price below cloud + Tenkan below Kijun + strong trend + volume spike
        short_condition = (
            price_below_cloud and 
            tenkan_below_kijun and 
            strong_trend and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
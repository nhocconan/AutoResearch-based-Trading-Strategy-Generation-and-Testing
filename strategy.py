#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Ichimoku_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 35 or len(df_1d) < 52:
        return np.zeros(n)
    
    # === 1d: Ichimoku Cloud (Tenkan, Kijun, Senkou A/B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 12h: ADX(14) for trend strength filter ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6h: Volume filter (current > 1.5x 24-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # Need enough data for Ichimoku (52) and ADX
    
    for i in range(start_idx, n):
        # Get aligned values
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        current_close = prices['close'].iloc[i]
        current_volume = volume[i]
        current_vol_ma = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or np.isnan(senkou_b) or
            np.isnan(adx_val) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Volume condition
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long conditions:
            # 1. Price above cloud (bullish)
            # 2. Tenkan > Kijun (bullish crossover)
            # 3. ADX > 25 (strong trend)
            # 4. Volume confirmation
            if (current_close > cloud_top and
                tenkan > kijun and
                adx_val > 25 and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below cloud (bearish)
            # 2. Tenkan < Kijun (bearish crossover)
            # 3. ADX > 25 (strong trend)
            # 4. Volume confirmation
            elif (current_close < cloud_bottom and
                  tenkan < kijun and
                  adx_val > 25 and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below cloud (trend change)
            # 2. Tenkan < Kijun (bearish crossover)
            # 3. ATR-based stop loss (using 6h ATR)
            if i >= 14:
                high_6h = prices['high'].values
                low_6h = prices['low'].values
                close_6h = prices['close'].values
                tr1 = np.abs(high_6h[1:] - low_6h[1:])
                tr2 = np.abs(high_6h[1:] - close_6h[:-1])
                tr3 = np.abs(low_6h[1:] - close_6h[:-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr = np.concatenate([[np.nan], tr])
                atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
                current_atr = atr_6h[i]
                if (current_close < cloud_bottom or
                    tenkan < kijun or
                    current_close < entry_price - 2.5 * current_atr):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above cloud (trend change)
            # 2. Tenkan > Kijun (bullish crossover)
            # 3. ATR-based stop loss
            if i >= 14:
                high_6h = prices['high'].values
                low_6h = prices['low'].values
                close_6h = prices['close'].values
                tr1 = np.abs(high_6h[1:] - low_6h[1:])
                tr2 = np.abs(high_6h[1:] - close_6h[:-1])
                tr3 = np.abs(low_6h[1:] - close_6h[:-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr = np.concatenate([[np.nan], tr])
                atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
                current_atr = atr_6h[i]
                if (current_close > cloud_top or
                    tenkan > kijun or
                    current_close > entry_price + 2.5 * current_atr):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals
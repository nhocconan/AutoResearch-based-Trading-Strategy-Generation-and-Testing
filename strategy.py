#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_ADX_Trend_v1
Hypothesis: On 6h timeframe, Ichimoku cloud from 1d (senkou span A/B) combined with ADX(14) > 25 trend strength filter and TK cross (tenkan/kijun) captures strong trending moves while avoiding ranging markets. 
In bullish regime (price > cloud), long on TK cross up; in bearish regime (price < cloud), short on TK cross down. 
ADX ensures we only trade when trend is strong, reducing whipsaw in choppy markets. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku components)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Ichimoku Cloud Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 6h ADX(14) for trend strength ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        adx_val = adx[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Price relative to cloud
        price_above_cloud = price > upper_cloud
        price_below_cloud = price < lower_cloud
        
        # TK Cross conditions
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            if price_above_cloud and strong_trend:
                # Bullish regime: long on TK cross up
                if tk_cross_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
            elif price_below_cloud and strong_trend:
                # Bearish regime: short on TK cross down
                if tk_cross_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions: TK cross in opposite direction or max hold time
            exit_signal = False
            if position == 1 and tk_cross_down:
                exit_signal = True
            elif position == -1 and tk_cross_up:
                exit_signal = True
            elif bars_since_entry >= max_hold_bars:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX_Trend_v1"
timeframe = "6h"
leverage = 1.0
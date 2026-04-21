#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h Ichimoku TK (Tenkan-Kijun) cross with 1d cloud filter for trend regime.
Long when TK crosses above AND price above 1d cloud (Senkou Span A/B).
Short when TK crosses below AND price below 1d cloud.
Uses ATR-based stop (2.5x) and minimum holding period of 3 bars to reduce churn.
Ichimoku cloud provides dynamic support/resistance that adapts to volatility,
working in both bull (cloud as support) and bear (cloud as resistance) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # === 1d Ichimoku Cloud Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_52 + min_low_52) / 2.0
    
    # Align 1d Ichimoku components to 6h timeframe (with proper delay for completed bars)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === 6h ATR (20-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        tenkan = tenkan_1d_aligned[i]
        kijun = kijun_1d_aligned[i]
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        
        # Cloud boundaries (top and bottom of Kumo)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # TK crossover detection (using previous bar values)
        if i > 100:
            prev_tenkan = tenkan_1d_aligned[i-1]
            prev_kijun = kijun_1d_aligned[i-1]
            tk_cross_above = (prev_tenkan <= prev_kijun) and (tenkan > kijun)
            tk_cross_below = (prev_tenkan >= prev_kijun) and (tenkan < kijun)
        else:
            tk_cross_above = False
            tk_cross_below = False
        
        if position == 0:
            # Long: TK cross above AND price above cloud
            long_condition = tk_cross_above and (price > cloud_top)
            # Short: TK cross below AND price below cloud
            short_condition = tk_cross_below and (price < cloud_bottom)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Cloud reversal exit (price falls below cloud)
                elif price < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Cloud reversal exit (price rises above cloud)
                elif price > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0
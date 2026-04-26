#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_12hTrend
Hypothesis: Ichimoku TK cross on 6h with price relative to 12h cloud (Senkou Span A/B) as trend filter. Works in bull/bear by requiring alignment with higher timeframe cloud direction. Targets 50-150 total trades over 4 years via TK cross frequency and cloud filter. Uses discrete sizing 0.25 to minimize fee churn. Includes ATR-based stoploss for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF cloud
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_12h = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_12h = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_12h = ((tenkan_12h + kijun_12h) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_12h = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for completed 12h bar)
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan_12h)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun_12h)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_12h, senkou_span_a_12h)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_12h, senkou_span_b_12h)
    
    # Calculate TK cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Determine cloud direction: price above/below cloud
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Load ATR for stoploss (using 6h ATR)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = np.abs(high_series - low_series)
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    tr1.iloc[0] = np.nan
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period median
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.3 * vol_median_20)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 14 for ATR, 20 for volume median
    start_idx = max(52, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or
            np.isnan(senkou_span_b_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr_6h[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: TK cross above AND price above cloud AND volume spike
            long_entry = tk_cross_above[i] and price_above_cloud[i] and vol_spike
            # Short: TK cross below AND price below cloud AND volume spike
            short_entry = tk_cross_below[i] and price_below_cloud[i] and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross below, ATR stoploss, or price below cloud
            stop_price = entry_price - 2.5 * atr_val
            if (tk_cross_below[i] or 
                close_val < stop_price or 
                close_val < cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TK cross above, ATR stoploss, or price above cloud
            stop_price = entry_price + 2.5 * atr_val
            if (tk_cross_above[i] or 
                close_val > stop_price or 
                close_val > cloud_top[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend"
timeframe = "6h"
leverage = 1.0
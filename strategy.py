#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeRegime
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price above/below 1d EMA50) and volume regime filter (ATR ratio > 1.2). Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B). Trend filter ensures alignment with higher timeframe momentum. Volume regime confirms institutional participation. Discrete sizing 0.25 limits trades (~20-40/year). Works in bull/bear via 1d trend filter and cloud breakout logic.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Cloud: between Senkou Span A and Senkou Span B
    # Upper cloud: max(Senkou Span A, Senkou Span B)
    # Lower cloud: min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (52 for Senkou Span B, 50 for ATR ratio and EMA)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.2  # volume regime
        size = fixed_size
        
        # Entry conditions: Ichimoku cloud breakout with volume spike AND aligned with 1d EMA50 trend
        # Long: price breaks above upper cloud (bullish breakout)
        # Short: price breaks below lower cloud (bearish breakout)
        long_entry = (close_val > upper_cloud_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < lower_cloud_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters cloud or trend reversal
            if close_val < upper_cloud_val and close_val > lower_cloud_val:  # back inside cloud
                signals[i] = 0.0
                position = 0
            elif close_val < ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters cloud or trend reversal
            if close_val > lower_cloud_val and close_val < upper_cloud_val:  # back inside cloud
                signals[i] = 0.0
                position = 0
            elif close_val > ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0
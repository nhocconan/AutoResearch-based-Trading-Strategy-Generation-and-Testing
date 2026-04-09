#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v1
# Hypothesis: 6h strategy using Ichimoku cloud for trend direction and entry timing, with 1d HTF EMA(50) for higher timeframe alignment. Long when price above cloud and Tenkan > Kijun; short when price below cloud and Tenkan < Kijun. Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear: Ichimoku adapts to volatility via cloud thickness, HTF EMA ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku Cloud components
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (high_series.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  low_series.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (high_series.rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 low_series.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = ((high_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      low_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Current cloud boundaries (already shifted for plotting)
    senkou_span_a_current = senkou_span_a.values
    senkou_span_b_current = senkou_span_b.values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Warmup: need enough data for Ichimoku calculations
    warmup = max(period_senkou_b + period_kijun, 50)  # 52 + 26 = 78
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(senkou_span_a_current[i]) or np.isnan(senkou_span_b_current[i]) or
            np.isnan(close[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries
        upper_cloud = max(senkou_span_a_current[i], senkou_span_b_current[i])
        lower_cloud = min(senkou_span_a_current[i], senkou_span_b_current[i])
        
        # Ichimoku signals
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        tenkan_above_kijun = tenkan_sen.iloc[i] > kijun_sen.iloc[i]
        tenkan_below_kijun = tenkan_sen.iloc[i] < kijun_sen.iloc[i]
        
        # HTF trend filter
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR Tenkan crosses below Kijun
            if close[i] < upper_cloud or tenkan_sen.iloc[i] < kijun_sen.iloc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR Tenkan crosses above Kijun
            if close[i] > lower_cloud or tenkan_sen.iloc[i] > kijun_sen.iloc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Ichimoku entry signals with HTF alignment
            bullish_setup = price_above_cloud and tenkan_above_kijun and htf_uptrend
            bearish_setup = price_below_cloud and tenkan_below_kijun and htf_downtrend
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals
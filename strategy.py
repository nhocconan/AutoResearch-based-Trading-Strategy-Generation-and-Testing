#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 12h Trend Filter
# Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B) from 6h timeframe
# Filters trades with 12h EMA50 trend direction to avoid counter-trend trades
# Long when price > cloud AND Tenkan > Kijun AND 12h EMA50 rising
# Short when price < cloud AND Tenkan < Kijun AND 12h EMA50 falling
# Ichimoku provides built-in support/resistance; EMA50 filter improves win rate in trends
# Target: 20-40 trades/year with clear trend-following logic

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Align Ichimoku components to 6h timeframe (no shift needed as calculations use historical data)
    tenkan_aligned = tenkan.values
    kijun_aligned = kijun.values
    senkou_span_a_aligned = senkou_span_a.values
    senkou_span_b_aligned = senkou_span_b.values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start from sufficient lookback
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + displacement
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Ichimoku signals
        price_above_close = close[i] > upper_cloud
        price_below_close = close[i] < lower_cloud
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # 12h EMA50 trend
        ema50_rising = ema50_12h_aligned[i] > ema50_12h_aligned[i-1]
        ema50_falling = ema50_12h_aligned[i] < ema50_12h_aligned[i-1]
        
        # Long: price above cloud, Tenkan > Kijun, 12h EMA50 rising
        if price_above_close and tenkan_above_kijun and ema50_rising and position <= 0:
            position = 1
            signals[i] = position_size
        # Short: price below cloud, Tenkan < Kijun, 12h EMA50 falling
        elif price_below_close and tenkan_below_kijun and ema50_falling and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit when Tenkan/Kijun cross reverses or price re-enters cloud
        elif position == 1 and (tenkan_below_kijun or close[i] < lower_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_above_kijun or close[i] > upper_cloud):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_12h_Ichimoku_EMA50_TrendFilter"
timeframe = "6h"
leverage = 1.0
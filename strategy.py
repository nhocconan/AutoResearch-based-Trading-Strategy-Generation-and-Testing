#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v2
# Hypothesis: 6h strategy using Ichimoku cloud (from 1d HTF) for trend direction, TK cross for entry timing, and volume confirmation (>1.3x 20-bar avg volume). Uses discrete sizing (0.25) to minimize fee churn. Ichimoku cloud provides strong trend filter that works in both bull/bear markets by avoiding trades against the higher timeframe trend. TK cross gives precise entry within the trend. Volume confirmation ensures breakout conviction. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear: cloud filters counter-trend trades, TK cross captures momentum in direction of trend, volume avoids false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    conversion_line = (high_1d_s.rolling(window=9, min_periods=9).max() + 
                      low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    conversion_line = conversion_line.values
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    base_line = (high_1d_s.rolling(window=26, min_periods=26).max() + 
                low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    base_line = base_line.values
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2 shifted 26 periods ahead
    leading_span_a = ((conversion_line + base_line) / 2)
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    highest_52 = high_1d_s.rolling(window=52, min_periods=52).max()
    lowest_52 = low_1d_s.rolling(window=52, min_periods=52).min()
    leading_span_b = ((highest_52 + lowest_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper shift for cloud)
    conversion_line_aligned = align_htf_to_ltf(prices, df_1d, conversion_line)
    base_line_aligned = align_htf_to_ltf(prices, df_1d, base_line)
    leading_span_a_aligned = align_htf_to_ltf(prices, df_1d, leading_span_a)
    leading_span_b_aligned = align_htf_to_ltf(prices, df_1d, leading_span_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(conversion_line_aligned[i]) or np.isnan(base_line_aligned[i]) or
            np.isnan(leading_span_a_aligned[i]) or np.isnan(leading_span_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Ichimoku trend: price above/both spans = uptrend, below/both spans = downtrend
        span_a = leading_span_a_aligned[i]
        span_b = leading_span_b_aligned[i]
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross: Conversion line crosses Base line
        tk_cross_up = (conversion_line_aligned[i] > base_line_aligned[i] and 
                      conversion_line_aligned[i-1] <= base_line_aligned[i-1])
        tk_cross_down = (conversion_line_aligned[i] < base_line_aligned[i] and 
                        conversion_line_aligned[i-1] >= base_line_aligned[i-1])
        
        if position == 1:  # Long position
            # Exit: price falls below cloud or TK cross down
            if price_below_cloud or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud or TK cross up
            if price_above_cloud or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for TK cross with volume and cloud filter
            bullish_setup = tk_cross_up and volume_confirmed and price_above_cloud
            bearish_setup = tk_cross_down and volume_confirmed and price_below_cloud
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals
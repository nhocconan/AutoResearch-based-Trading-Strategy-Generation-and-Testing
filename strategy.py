#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using 1d Ichimoku cloud as trend filter and 6h price breakout from cloud boundaries.
# In trending markets, price breaks above/below cloud with momentum; in ranging markets, cloud acts as support/resistance.
# Uses 6h Donchian(20) breakout for entry timing, filtered by 1d Ichimoku cloud color and price position relative to cloud.
# Volume confirmation ensures breakout validity. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years by requiring cloud alignment + breakout + volume.
# Primary timeframe: 6h, HTF: 1d for Ichimoku cloud.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_v1"
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
    
    # 1d HTF data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for cloud calculation but needed for alignment
    
    # Cloud boundaries: Senkou Span A and B, shifted forward 26 periods
    # For HTF alignment, we need the current cloud (values from 26 periods ago)
    # So we use unshifted Senkou Span A/B and align normally (align_htf_to_ltf handles completed bar delay)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Cloud color: green (bullish) when Span A > Span B, red (bearish) when Span A < Span B
    cloud_bullish = senkou_span_a_aligned > senkou_span_b_aligned
    
    # 6h Donchian(20) for breakout detection
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below cloud bottom or volume dries up
            if close[i] < cloud_bottom[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above cloud top or volume dries up
            if close[i] > cloud_top[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above cloud top in bullish cloud OR price above cloud in bearish cloud with strong breakout
                if close[i] > cloud_top[i] and high[i] > cloud_top[i]:
                    # Stronger signal when cloud is bullish
                    if cloud_bullish[i]:
                        position = 1
                        signals[i] = 0.25
                    # In bearish cloud, require price to be significantly above cloud (avoid false breakouts)
                    elif close[i] > cloud_top[i] * 1.01:  # 1% above cloud top
                        position = 1
                        signals[i] = 0.25
                # Short entry: price breaks below cloud bottom in bearish cloud OR price below cloud in bullish cloud with strong breakout
                elif close[i] < cloud_bottom[i] and low[i] < cloud_bottom[i]:
                    # Stronger signal when cloud is bearish
                    if not cloud_bullish[i]:
                        position = -1
                        signals[i] = -0.25
                    # In bullish cloud, require price to be significantly below cloud (avoid false breakouts)
                    elif close[i] < cloud_bottom[i] * 0.99:  # 1% below cloud bottom
                        position = -1
                        signals[i] = -0.25
    
    return signals
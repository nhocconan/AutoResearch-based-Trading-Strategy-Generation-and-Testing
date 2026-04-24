#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for Ichimoku cloud and trend alignment.
- Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displacement).
- Entry logic: Long when price closes above Senkou Span A (cloud top) with bullish TK cross (Tenkan > Kijun) and volume spike.
               Short when price closes below Senkou Span B (cloud bottom) with bearish TK cross (Tenkan < Kijun) and volume spike.
- Trend filter: Only long when 6h close > 1d Kijun-sen (26-period), only short when 6h close < 1d Kijun-sen.
- Volume confirmation: Current 6h volume > 1.8 * 20-period 6h volume MA (moderate to balance signal quality and frequency).
- Discrete signal size: 0.25 to manage drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Ichimoku cloud acts as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (no additional delay needed for completed cloud)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Trend filter: 6h close vs 1d Kijun-sen (26-period)
    uptrend = close > kijun_sen_aligned
    downtrend = close < kijun_sen_aligned
    
    # TK cross: Tenkan-sen > Kijun-sen (bullish), Tenkan-sen < Kijun-sen (bearish)
    tk_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52)  # Need sufficient data for Ichimoku (52-period) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above Senkou Span A (cloud top) AND bullish TK cross AND uptrend AND volume spike
            if (close[i] > senkou_span_a_aligned[i] and tk_bullish[i] and 
                uptrend[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price closes below Senkou Span B (cloud bottom) AND bearish TK cross AND downtrend AND volume spike
            elif (close[i] < senkou_span_b_aligned[i] and tk_bearish[i] and 
                  downtrend[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Senkou Span A (cloud top) or bearish TK cross
            if close[i] <= senkou_span_a_aligned[i] or not tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Senkou Span B (cloud bottom) or bullish TK cross
            if close[i] >= senkou_span_b_aligned[i] or tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
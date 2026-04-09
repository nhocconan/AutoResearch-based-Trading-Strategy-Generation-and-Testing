#!/usr/bin/env python3
# 6h_ichimoku_cloud_regime_v1
# Hypothesis: 6h strategy using Ichimoku Cloud from 1d timeframe for trend/filter,
# with TK cross on 6h for entry timing. Uses volume confirmation and discrete sizing.
# Works in bull/bear markets: Ichimoku cloud acts as dynamic S/R and regime filter
# (price above cloud = bull bias, below = bear bias), TK cross provides momentum
# entries with volume validation. Multi-timeframe alignment ensures no look-ahead.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_regime_v1"
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
    
    # Ichimoku Cloud calculation on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Conversion line (Tenkan-sen): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    tenkan_sen = ((high_1d_s.rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                   low_1d_s.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2).values
    
    # Base line (Kijun-sen): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = ((high_1d_s.rolling(window=period_kijun, min_periods=period_kijun).max() +
                  low_1d_s.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2).values
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((high_1d_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                      low_1d_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Lagging Span (Chikou Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (completed 1d bars only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Ichimoku trend filters
        price_above_cloud = (close[i] > senkou_span_a_aligned[i] and 
                            close[i] > senkou_span_b_aligned[i])
        price_below_cloud = (close[i] < senkou_span_a_aligned[i] and 
                            close[i] < senkou_span_b_aligned[i])
        
        # TK Cross signals
        tk_bullish_cross = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                           tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_bearish_cross = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                           tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        if position == 1:  # Long position
            # Exit: price breaks below cloud OR TK bearish cross
            if price_below_cloud or tk_bearish_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above cloud OR TK bullish cross
            if price_above_cloud or tk_bullish_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud + TK bullish cross + volume
            if price_above_cloud and tk_bullish_cross and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud + TK bearish cross + volume
            elif price_below_cloud and tk_bearish_cross and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals
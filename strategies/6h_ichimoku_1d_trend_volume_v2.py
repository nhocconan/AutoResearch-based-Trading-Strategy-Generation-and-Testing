#!/usr/bin/env python3
# 6h_ichimoku_1d_trend_volume_v2
# Hypothesis: 6h strategy using 1d Ichimoku cloud (Senkou Span A/B) for trend filter, TK cross for entry timing, and volume confirmation.
# In bull markets: price above cloud + TK cross bullish + volume spike → long
# In bear markets: price below cloud + TK cross bearish + volume spike → short
# The cloud acts as dynamic support/resistance, reducing whipsaws. Volume > 1.5x 20-period average filters weak moves.
# Discrete sizing (±0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v2"
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
    
    # 1d HTF data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe (with displacement for cloud)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=displacement)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=displacement)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud (trend reversal)
            if close[i] < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud (trend reversal)
            if close[i] > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # TK cross: Tenkan-sen crossing above Kijun-sen (bullish)
                tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
                # TK cross: Tenkan-sen crossing below Kijun-sen (bearish)
                tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
                
                # Long: price above cloud + bullish TK cross
                if close[i] > upper_cloud and tk_cross_bullish:
                    position = 1
                    signals[i] = 0.25
                # Short: price below cloud + bearish TK cross
                elif close[i] < lower_cloud and tk_cross_bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals
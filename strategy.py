#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_ichimoku_trend_v1
# Ichimoku Cloud system on 1-day chart for trend direction, with 6h entries on pullbacks to Kumo.
# Works in bull markets by buying dips in uptrend (price above cloud), and in bear markets
# by selling rallies in downtrend (price below cloud). Uses TK cross for entry timing and
# Kijun as dynamic support/resistance. Target: 15-30 trades/year per symbol.
name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard periods: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    # Not used for signals but needed for cloud calculation
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after warmup for Ichimoku
        # Skip if Ichimoku data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above/below cloud
        # Cloud top = max(Senkou Span A, Senkou Span B)
        # Cloud bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Bullish trend: price above cloud
        # Bearish trend: price below cloud
        # Neutral: price inside cloud (avoid trading)
        if close[i] > cloud_top:
            # Uptrend: look for long entries on pullbacks
            # Enter long when price touches Kijun-sen (support) and Tenkan > Kijun (bullish cross)
            if (close[i] <= kijun_sen_aligned[i] * 1.005 and  # near Kijun support
                tenkan_sen_aligned[i] > kijun_sen_aligned[i]):  # bullish TK cross
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # hold long
            # Exit long when price breaks below Kijun or TK cross turns bearish
            elif (position == 1 and 
                  (close[i] < kijun_sen_aligned[i] * 0.995 or  # below Kijun
                   tenkan_sen_aligned[i] < kijun_sen_aligned[i])):  # bearish TK cross
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else 0.0  # hold or flat
                
        elif close[i] < cloud_bottom:
            # Downtrend: look for short entries on rallies
            # Enter short when price touches Kijun-sen (resistance) and Tenkan < Kijun (bearish cross)
            if (close[i] >= kijun_sen_aligned[i] * 0.995 and  # near Kijun resistance
                tenkan_sen_aligned[i] < kijun_sen_aligned[i]):  # bearish TK cross
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # hold short
            # Exit short when price breaks above Kijun or TK cross turns bullish
            elif (position == -1 and 
                  (close[i] > kijun_sen_aligned[i] * 1.005 or  # above Kijun
                   tenkan_sen_aligned[i] > kijun_sen_aligned[i])):  # bullish TK cross
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25 if position == -1 else 0.0  # hold or flat
        else:
            # Price inside cloud: stay flat
            position = 0
            signals[i] = 0.0
    
    return signals
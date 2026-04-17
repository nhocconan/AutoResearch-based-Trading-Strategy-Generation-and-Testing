#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses 1d Tenkan-sen and Kijun-sen cross for entry, 1d Kumo (cloud) for trend filter.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 12-30 trades/year to stay within optimal range for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or 
            np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Ichimoku signals
        # TK Cross: Tenkan-sen crosses above/below Kijun-sen
        tk_cross_up = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_down = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        # Cloud: price above/below both Senkou spans
        cloud_top = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross up + price above cloud + volume filter
            if (tk_cross_up and price_above_cloud and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + volume filter
            elif (tk_cross_down and price_below_cloud and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price below cloud
            if (tk_cross_down) or (close[i] < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price above cloud
            if (tk_cross_up) or (close[i] > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dTK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0
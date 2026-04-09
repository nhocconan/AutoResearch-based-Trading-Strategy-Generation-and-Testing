#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with volume confirmation
# Ichimoku provides multi-component trend/cloud structure (Tenkan, Kijun, Senkou A/B, Chikou)
# Price above/below cloud determines trend bias; TK cross provides entry timing
# Volume confirmation (current 6h volume > 2.0x 20-period average) filters false signals
# Works in bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "6h_1d_ichimoku_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average 6h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit when price closes below cloud (trend reversal)
            if close[i] < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price closes above cloud (trend reversal)
            if close[i] > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter on TK cross with volume confirmation and cloud filter
            # Bullish TK cross: Tenkan crosses above Kijun
            # Bearish TK cross: Tenkan crosses below Kijun
            if i > 100:  # Need previous values for crossover
                tenkan_prev = tenkan_aligned[i-1]
                kijun_prev = kijun_aligned[i-1]
                tenkan_curr = tenkan_aligned[i]
                kijun_curr = kijun_aligned[i]
                
                bullish_tk = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
                bearish_tk = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
                
                if volume_confirmed:
                    if bullish_tk and close[i] > upper_cloud:
                        # Strong bullish signal: price above cloud + bullish TK cross
                        position = 1
                        signals[i] = 0.25
                    elif bearish_tk and close[i] < lower_cloud:
                        # Strong bearish signal: price below cloud + bearish TK cross
                        position = -1
                        signals[i] = -0.25
    
    return signals
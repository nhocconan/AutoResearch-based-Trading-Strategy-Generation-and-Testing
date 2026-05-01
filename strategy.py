#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d TK Cross and Volume Spike Confirmation
# Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B, Chikou) from 1d timeframe
# Trades when price is above/below cloud with TK cross alignment and volume confirmation
# Works in bull/bear by trading with cloud direction and avoiding range/chop markets
# Target: 12-37 trades/year (50-150 over 4 years) via strict confluence requirements

name = "6h_Ichimoku_Cloud_1dTKCross_VolumeSpike_v1"
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
    
    # 1d HTF data for Ichimoku calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou = close_1d  # Will be aligned with proper delay
    
    # Align all Ichimoku components to 6h timeframe
    # Note: Senkou spans are already shifted in calculation, so we align the raw values
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou, additional_delay_bars=26)  # Chikou needs 26-bar delay
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK Cross signals
    tk_cross_bull = tenkan_aligned > kijun_aligned  # Tenkan above Kijun
    tk_cross_bear = tenkan_aligned < kijun_aligned  # Tenkan below Kijun
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (52 for Senkou B, 26 for Chikou alignment, 20 for volume)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(chikou_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Avoid whipsaw in ranging markets: require price to be outside cloud
        in_cloud = (close[i] >= cloud_bottom[i]) & (close[i] <= cloud_top[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above cloud, TK cross bull, volume spike, not in cloud
            if price_above_cloud[i] and tk_cross_bull[i] and volume_spike[i] and not in_cloud:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, TK cross bear, volume spike, not in cloud
            elif price_below_cloud[i] and tk_cross_bear[i] and volume_spike[i] and not in_cloud:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price falls below cloud or TK cross turns bearish
            if price_below_cloud[i] or not tk_cross_bull[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud or TK cross turns bullish
            if price_above_cloud[i] or not tk_cross_bear[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_ichimoku_kijun_sen_cross_v1
# Hypothesis: 6h strategy using Ichimoku Kijun-sen (base line) cross of Tenkan-sen (conversion line) with cloud filter from 1d HTF. Enters long when Tenkan crosses above Kijun with price above 1d cloud; enters short when Tenkan crosses below Kijun with price below 1d cloud. Uses volume confirmation (>1.5x 20-bar avg) to filter false signals. Exits on opposite cross or close beyond cloud edge. Weekly trend filter via price vs 1w EMA(20) avoids counter-trend trades. Designed for low turnover (12-37 trades/year) with discrete sizing (0.25) to minimize fee drag. Works in bull/bear via cloud as dynamic support/resistance and multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kijun_sen_cross_v1"
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
    
    # Multi-timeframe: 1d Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    tenkan_1d = (high_1d_s.rolling(window=9, min_periods=9).mean() + 
                 low_1d_s.rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (high_1d_s.rolling(window=26, min_periods=26).mean() + 
                low_1d_s.rolling(window=26, min_periods=26).mean()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1d = ((high_1d_s.rolling(window=52, min_periods=52).mean() + 
                    low_1d_s.rolling(window=52, min_periods=52).mean()) / 2).shift(26)
    
    # Align 1d Ichimoku to 6h timeframe (wait for 1d close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Cloud boundaries (top/bottom of cloud)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Multi-timeframe: 1w EMA(20) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h Ichimoku components (for entry signals)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    tenkan_6h = (high_s.rolling(window=9, min_periods=9).mean() + 
                 low_s.rolling(window=9, min_periods=9).mean()) / 2
    kijun_6h = (high_s.rolling(window=26, min_periods=26).mean() + 
                low_s.rolling(window=26, min_periods=26).mean()) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        price_above_1w_ema = close[i] > ema_20_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_20_1w_aligned[i]
        
        # Ichimoku cross signals (6h timeframe)
        tenkan_prev = tenkan_6h[i-1]
        kijun_prev = kijun_6h[i-1]
        tenkan_cross_above = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_prev <= kijun_prev)
        tenkan_cross_below = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_prev >= kijun_prev)
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price breaks below cloud bottom
            if tenkan_cross_below or close[i] < cloud_bottom[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price breaks above cloud top
            if tenkan_cross_above or close[i] > cloud_top[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Ichimoku cross with volume, cloud filter, and 1w trend alignment
            bullish_setup = (tenkan_cross_above and 
                           volume_confirmed and 
                           close[i] > cloud_top[i] and 
                           price_above_1w_ema)
            bearish_setup = (tenkan_cross_below and 
                           volume_confirmed and 
                           close[i] < cloud_bottom[i] and 
                           price_below_1w_ema)
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals
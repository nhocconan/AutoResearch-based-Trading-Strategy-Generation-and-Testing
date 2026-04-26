#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1dTrendFilter_VolumeSpike
Hypothesis: Ichimoku cloud breakout with 1d EMA50 trend filter and volume confirmation on 6h timeframe. 
Uses Senkou Span A/B cloud from daily Ichimoku for major trend direction, Tenkan/Kijun cross for entry timing, 
and volume spike for confirmation. Designed for low trade frequency (20-50/year) to minimize fee drag 
while capturing strong trending moves in both bull and bear markets. The cloud acts as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (completed 1d bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x average volume on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku (52), EMA50 (50), volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Ichimoku signals
        # Bullish: Tenkan crosses above Kijun AND price above cloud
        tk_bullish_cross = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        price_above_cloud = close_val > upper_cloud
        
        # Bearish: Tenkan crosses below Kijun AND price below cloud
        tk_bearish_cross = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        price_below_cloud = close_val < lower_cloud
        
        if position == 0:
            # Long: bullish TK cross above cloud with volume confirmation and uptrend
            long_signal = tk_bullish_cross and price_above_cloud and (volume_val > 1.8 * vol_ma_val) and uptrend
            # Short: bearish TK cross below cloud with volume confirmation and downtrend
            short_signal = tk_bearish_cross and price_below_cloud and (volume_val > 1.8 * vol_ma_val) and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below cloud OR trend reversal
            if (close_val < lower_cloud) or (not uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud OR trend reversal
            if (close_val > upper_cloud) or (not downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0
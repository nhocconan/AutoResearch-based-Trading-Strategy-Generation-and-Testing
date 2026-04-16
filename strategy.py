#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud (TK cross + price above/below cloud) with 1w volume confirmation.
# Long when: TK cross bullish, price above cloud, volume > 1.3x 20-period 1w average.
# Short when: TK cross bearish, price below cloud, volume > 1.3x 20-period 1w average.
# Exit when: TK cross reverses OR price crosses cloud midpoint (Senkou Span A/B average).
# Uses discrete position size 0.25. Designed to capture medium-term trends with trend and volume confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Ichimoku Cloud ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Cloud boundaries: upper = max(Span A, Span B), lower = min(Span A, Span B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    cloud_mid = (cloud_top + cloud_bottom) / 2  # Midpoint for exit
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    cloud_mid_aligned = align_htf_to_ltf(prices, df_1d, cloud_mid)
    
    # === 1w Indicators: Volume Spike ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.3 * vol_ma_1w_aligned)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all Ichimoku components are valid (max 52 periods)
    warmup = 100
    
    # Track position state and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(cloud_mid_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # TK cross conditions
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price vs cloud
        price_above_cloud = price > cloud_top_aligned[i]
        price_below_cloud = price < cloud_bottom_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if TK cross turns bearish OR price crosses below cloud midpoint
            if not tk_bullish or price < cloud_mid_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if TK cross turns bullish OR price crosses above cloud midpoint
            if not tk_bearish or price > cloud_mid_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish TK cross, price above cloud, volume spike
            if tk_bullish and price_above_cloud and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bearish TK cross, price below cloud, volume spike
            elif tk_bearish and price_below_cloud and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTK_Cloud_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0
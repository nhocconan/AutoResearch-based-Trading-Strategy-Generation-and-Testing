#!/usr/bin/env python3
# 6h_RVOL_Ichimoku_Trend
# Hypothesis: Combines Ichimoku cloud for trend direction with relative volume (RVOL) spikes on 6h timeframe.
# Long when: price above Ichimoku cloud, Tenkan > Kijun, and RVOL > 1.8 (strong bullish momentum).
# Short when: price below Ichimoku cloud, Tenkan < Kijun, and RVOL > 1.8 (strong bearish momentum).
# Exit when price crosses back into/under the cloud or Tenkan/Kijun cross reverses.
# Ichimoku provides multi-line trend/filter system; RVOL ensures participation in high-volume moves.
# Works in bull markets by catching strong uptrends and in bear markets by catching strong downtrends.
# Volume filter reduces false signals in low-volatility environments.

name = "6h_RVOL_Ichimoku_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku (higher timeframe for trend context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Ichimoku Cloud calculation on 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = np.full_like(high_1d, np.nan)
    min_low_tenkan = np.full_like(low_1d, np.nan)
    for i in range(period_tenkan-1, len(high_1d)):
        max_high_tenkan[i] = np.max(high_1d[i-(period_tenkan-1):i+1])
        min_low_tenkan[i] = np.min(low_1d[i-(period_tenkan-1):i+1])
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = np.full_like(high_1d, np.nan)
    min_low_kijun = np.full_like(low_1d, np.nan)
    for i in range(period_kijun-1, len(high_1d)):
        max_high_kijun[i] = np.max(high_1d[i-(period_kijun-1):i+1])
        min_low_kijun[i] = np.min(low_1d[i-(period_kijun-1):i+1])
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = np.full_like(high_1d, np.nan)
    min_low_senkou_b = np.full_like(low_1d, np.nan)
    for i in range(period_senkou_b-1, len(high_1d)):
        max_high_senkou_b[i] = np.max(high_1d[i-(period_senkou_b-1):i+1])
        min_low_senkou_b[i] = np.min(low_1d[i-(period_senkou_b-1):i+1])
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # --- Relative Volume (RVOL) on 6h: current volume / 20-period average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    rvol = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma[i] > 0:
            rvol[i] = volume[i] / vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Ichimoku (52 periods) and RVOL (20 periods)
    start_idx = 52  # Ichimoku needs 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku trend conditions
        price_above_cloud = (close[i] > senkou_a_aligned[i]) and (close[i] > senkou_b_aligned[i])
        price_below_cloud = (close[i] < senkou_a_aligned[i]) and (close[i] < senkou_b_aligned[i])
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Volume condition: RVOL > 1.8 (significant volume participation)
        vol_condition = rvol[i] > 1.8
        
        if position == 0:
            if price_above_cloud and tenkan_above_kijun and vol_condition:
                # Long: bullish Ichimoku + volume participation
                signals[i] = 0.25
                position = 1
            elif price_below_cloud and tenkan_below_kijun and vol_condition:
                # Short: bearish Ichimoku + volume participation
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price enters cloud OR Tenkan/Kijun cross turns bearish
                if (not price_above_cloud) or (not tenkan_above_kijun):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price exits cloud OR Tenkan/Kijun cross turns bullish
                if (not price_below_cloud) or (not tenkan_below_kijun):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
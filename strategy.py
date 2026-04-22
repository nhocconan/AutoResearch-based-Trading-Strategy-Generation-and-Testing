#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data
# Trend filter: price above/below 1d Kumo cloud (Senkou Span A/B)
# Entry: TK cross (Tenkan crosses Kijun) in direction of 1d trend with volume confirmation
# Exit: TK cross in opposite direction or price crosses Kumo
# Ichimoku provides built-in support/resistance and trend strength
# Designed for 6h timeframe to target 15-30 trades/year per symbol.
# Works in bull/bear markets via trend filter reducing whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (Kumo cloud) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    # Not used for signals to avoid look-ahead
    
    # Calculate 1d Kumo cloud (Senkou Span A/B) for trend filter
    # These are plotted 26 periods ahead, so we need historical values
    # For 1d data, calculate Senkou Span A/B then shift back 26 periods to align
    max_high_1d_tenkan = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_1d_tenkan = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_1d_tenkan + min_low_1d_tenkan) / 2
    
    max_high_1d_kijun = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_1d_kijun = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_1d_kijun + min_low_1d_kijun) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    max_high_1d_senkou_b = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_1d_senkou_b = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_1d_senkou_b + min_low_1d_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Using 1d df for alignment but 6h values
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Align 1d Kumo components (need to use actual 1d data for alignment)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Kumo cloud boundaries (Senkou Span A/B) - note: these are already plotted ahead
    # For trend filter, we use current Kumo (no additional shift needed as align_htf_to_ltf handles it)
    kumo_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    kumo_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # TK Cross signals
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            # Long: TK cross up + price above Kumo + volume spike
            if (tk_cross_up and close[i] > kumo_top[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below Kumo + volume spike
            elif (tk_cross_down and close[i] < kumo_bottom[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: TK cross down OR price drops below Kumo
                tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
                price_below_kumo = close[i] < kumo_bottom[i]
                if tk_cross_down or price_below_kumo:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit: TK cross up OR price rises above Kumo
                tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
                price_above_kumo = close[i] > kumo_top[i]
                if tk_cross_up or price_above_kumo:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dKumoTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day TK cross and volume confirmation
# Ichimoku provides multi-layered support/resistance (cloud), momentum (TK cross),
# and trend direction (price vs cloud). Using daily TK cross for signal direction
# and 6h price vs daily cloud for entry/exit reduces whipsaw. Volume confirmation
# ensures institutional participation. Designed for 6h to achieve 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-day Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Align to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values, additional_delay_bars=26)
    
    # TK Cross signal: 1 when Tenkan > Kijun, -1 when Tenkan < Kijun
    tk_cross = np.where(tenkan_sen_aligned > kijun_sen_aligned, 1, -1)
    
    # Cloud top and bottom (aligned)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # === 6-day Volume Spike (vs 20-period average) ===
    df_6d = get_htf_data(prices, '6d')
    volume_6d = df_6d['volume'].values
    vol_ma_20_6d = pd.Series(volume_6d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6d_aligned = align_htf_to_ltf(prices, df_6d, vol_ma_20_6d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma_20_6d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 6h price and volume
        price_6d_aligned = align_htf_to_ltf(prices, df_6d, df_6d['close'].values)
        volume_6d_aligned = align_htf_to_ltf(prices, df_6d, volume_6d)
        
        # Volume spike: current 6d volume > 1.5x 20-period average
        vol_spike = volume_6d_aligned[i] > vol_ma_20_6d_aligned[i] * 1.5
        
        # Price relative to cloud
        price_above_cloud = price_6d_aligned[i] > cloud_top[i]
        price_below_cloud = price_6d_aligned[i] < cloud_bottom[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike:
                # Long: price above cloud + bullish TK cross
                if price_above_cloud and tk_cross[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: price below cloud + bearish TK cross
                elif price_below_cloud and tk_cross[i] == -1:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long if price enters cloud or TK cross turns bearish
            if not price_above_cloud or tk_cross[i] == -1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price enters cloud or TK cross turns bullish
            if not price_below_cloud or tk_cross[i] == 1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dTKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0
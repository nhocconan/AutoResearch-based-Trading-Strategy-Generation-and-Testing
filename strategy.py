#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation
# - Ichimoku (9,26,52) from 6h: Tenkan/Kijun cross + price above/below cloud for entry
# - 1w ADX(14) > 20 to ensure weekly trend alignment and avoid counter-trend trades
# - Volume confirmation: current 6h volume > 1.5x 24-period average (4 days)
# - Designed for 6h timeframe: targets 12-30 trades/year to avoid fee drag
# - Works in bull/bear markets: weekly ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "6h_1w_ichimoku_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 6h Ichimoku components (9,26,52)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components (no extra delay needed as they are based on completed periods)
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high_6h, 'low': low_6h, 'close': close_6h}), senkou_b)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_24 = pd.Series(volume_6h).rolling(window=24, min_periods=24).mean().values  # 4 days
    vol_spike = volume_6h > (1.5 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR Tenkan/Kijun death cross
            if close_6h[i] < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR Tenkan/Kijun golden cross
            if close_6h[i] > cloud_top or tenkan_aligned[i] > kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Ichimoku signals with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 20:
                # Golden cross: Tenkan crosses above Kijun + price above cloud
                if (tenkan_aligned[i] > kijun_aligned[i] and 
                    close_6h[i] > cloud_top):
                    position = 1
                    signals[i] = 0.25
                # Death cross: Tenkan crosses below Kijun + price below cloud
                elif (tenkan_aligned[i] < kijun_aligned[i] and 
                      close_6h[i] < cloud_bottom):
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike
Hypothesis: Ichimoku cloud (TK cross + price vs cloud) from 6h, filtered by 1d EMA50 trend and volume spike (>2.0x 20-bar MA). Uses Ichimoku as a proven trend/momentum indicator (Tier 8 in program.md) with 1d trend alignment to avoid counter-trend trades and volume confirmation to reduce false signals. Designed for 12-37 trades/year (50-150 total over 4 years) on 6h timeframe to minimize fee drag. Works in bull/bear markets by following 1d trend while using Ichimoku cloud for structure-based entries and exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h data (Tenkan-sen, Kijun-sen, Senkou Span A/B)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it's lagging
    
    # Align Ichimoku components to 6h timeframe (they are already on 6h)
    # No alignment needed as we calculated on 6h data directly
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (52 for senkou b, 20 for vol)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Ichimoku signals:
        # Bullish: Tenkan > Kijun AND price > Senkou Span (top of cloud)
        # Bearish: Tenkan < Kijun AND price < Senkou Span (bottom of cloud)
        # Cloud top = max(senkou_a, senkou_b), Cloud bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        bullish_ichimoku = (tenkan_val > kijun_val) and (close_val > cloud_top)
        bearish_ichimoku = (tenkan_val < kijun_val) and (close_val < cloud_bottom)
        
        # Entry conditions: Ichimoku signal in 1d trend direction with volume spike
        long_entry = bullish_ichimoku and bullish_1d and vol_spike
        short_entry = bearish_ichimoku and bearish_1d and vol_spike
        
        # Exit conditions: opposite Ichimoku signal (tenkan/kijun cross reverse)
        exit_long = tenkan_val < kijun_val  # Tenkan crosses below Kijun
        exit_short = tenkan_val > kijun_val  # Tenkan crosses above Kijun
        
        # Minimum holding period: 3 bars (to avoid whipsaw)
        min_hold = 3
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
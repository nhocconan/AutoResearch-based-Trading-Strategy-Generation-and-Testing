#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike
Hypothesis: Ichimoku cloud breakout with 12h trend filter (price > Kumo) and volume confirmation (>1.5x 20-bar MA). Uses the cloud as dynamic support/resistance and TK cross for momentum confirmation. Works in bull/bear markets by following 12h trend while using Ichimoku structure for precise entries. Volume spike reduces false breakouts. Target: 15-25 trades/year.
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
    
    # Load 12h data ONCE before loop for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h close for trend filter
    close_12h = df_12h['close'].values
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for entry/exit as it requires future data
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # Same timeframe, no alignment needed but using helper for consistency
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, prices, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, prices, cloud_bottom)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (26 for Ichimoku, 20 for volume)
    start_idx = max(26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or 
            np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(trend_12h_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        vol_spike = volume_spike[i]
        trend_12h_val = trend_12h_aligned[i]
        
        # Determine 12h trend: bullish if close > 12h close, bearish if close < 12h close
        bullish_12h = close_val > trend_12h_val
        bearish_12h = close_val < trend_12h_val
        
        # TK cross: Tenkan > Kijun = bullish momentum, Tenkan < Kijun = bearish momentum
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price relative to cloud: above cloud = bullish, below cloud = bearish, inside cloud = neutral
        price_above_cloud = close_val > cloud_top_val
        price_below_cloud = close_val < cloud_bottom_val
        
        # Entry conditions: 
        # Long: price breaks above cloud + TK bullish + 12h bullish trend + volume spike
        # Short: price breaks below cloud + TK bearish + 12h bearish trend + volume spike
        long_entry = price_above_cloud and tk_bullish and bullish_12h and vol_spike
        short_entry = price_below_cloud and tk_bearish and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price falls below cloud or TK turns bearish
            if close_val < cloud_top_val or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when price rises above cloud or TK turns bullish
            if close_val > cloud_bottom_val or not tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v3
Hypothesis: Trade 6h Ichimoku cloud breaks with 1d trend filter and volume confirmation.
The Ichimoku cloud acts as dynamic support/resistance. In bull markets, price above cloud with TK cross up signals long;
in bear markets, price below cloud with TK cross down signals short. 1d trend filter ensures we trade with higher timeframe momentum,
reducing whipsaws. Volume confirmation adds conviction. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it's purely lagging
    
    # Align Ichimoku components to 6h (they are already on 6h)
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK cross
    tk_cross_up = tenkan > kijun
    tk_cross_down = tenkan < kijun
    
    # Volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), EMA(50), volume MA
    start_idx = max(52, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        price_above_cloud = close_val > cloud_top[i]
        price_below_cloud = close_val < cloud_bottom[i]
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above cloud AND TK cross up AND 1d trend up AND volume spike
            long_signal = price_above_cloud and tk_cross_up[i] and trend_1d_up and vol_spike
            
            # Short: price below cloud AND TK cross down AND 1d trend down AND volume spike
            short_signal = price_below_cloud and tk_cross_down[i] and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR TK cross down OR 1d trend flips down
            if (not price_above_cloud) or (not tk_cross_up[i]) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK cross up OR 1d trend flips up
            if (not price_below_cloud) or (not tk_cross_down[i]) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v3"
timeframe = "6h"
leverage = 1.0
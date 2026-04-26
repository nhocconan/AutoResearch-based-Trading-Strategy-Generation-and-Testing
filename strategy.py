#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price > EMA50) and volume confirmation (2.0x). 
Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross.
Works in bull/bear via 1d trend alignment: only long in uptrend, short in downtrend.
Targets 12-30 trades/year by requiring confluence of cloud breakout, TK cross, trend, and volume.
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Cloud top/bottom (Senkou Span A/B are plotted 26 periods ahead)
    # For cloud at time i, we use values from i-26
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_up = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_cross_down = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # Volume filter: volume > 2.0 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 20 for volume MA)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku cloud breakout with trend and volume filters
        if position == 0:
            # Long: Price breaks above cloud AND TK cross up AND 1d uptrend AND volume spike
            if (close[i] > cloud_top[i] and tk_cross_up[i] and 
                trend_1d[i] == 1 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND TK cross down AND 1d downtrend AND volume spike
            elif (close[i] < cloud_bottom[i] and tk_cross_down[i] and 
                  trend_1d[i] == -1 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below cloud OR TK cross down OR 1d trend turns down
            if (close[i] < cloud_bottom[i] or tk_cross_down[i] or trend_1d[i] == -1):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above cloud OR TK cross up OR 1d trend turns up
            if (close[i] > cloud_top[i] or tk_cross_up[i] or trend_1d[i] == 1):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
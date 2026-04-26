#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter_v1
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) on 6h as early trend change signal, filtered by 1d EMA50 trend direction to avoid whipsaws. Only take longs when price > Kumo and 1d EMA50 uptrend, shorts when price < Kumo and 1d EMA50 downtrend. Uses volume confirmation (1.5x 20-period average) to ensure breakout strength. Designed for 6h timeframe to capture medium-term swings in BTC/ETH with discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) by requiring confluence of Kumo twist, 1d trend filter, and volume spike.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
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
    
    # Kumo (cloud) top and bottom
    kumO_top = np.maximum(senkou_a, senkou_b)
    kumO_bottom = np.minimum(senkou_a, senkou_b)
    
    # Kumo twist: Senkou A crosses Senkou B (trend change signal)
    # Twist up: Senkou A crosses above Senkou B
    # Twist down: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = senkou_a[0]  # avoid NaN on first element
    senkou_b_prev[0] = senkou_b[0]
    
    twist_up = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    twist_down = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Senkou A, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(kumO_top[i]) or np.isnan(kumO_bottom[i]) or
            np.isnan(twist_up[i]) or np.isnan(twist_down[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumO_top[i]
        price_below_kumo = close[i] < kumO_bottom[i]
        
        # Entry logic: Kumo twist + 1d trend filter + volume spike + price position
        if htf_trend[i] == 1:  # 1d uptrend
            # Long on Kumo twist up + price above Kumo + volume spike
            if twist_up[i] and price_above_kumo and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: price falls below Kumo bottom
            elif position == 1 and price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # 1d downtrend
            # Short on Kumo twist down + price below Kumo + volume spike
            if twist_down[i] and price_below_kumo and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: price rises above Kumo top
            elif position == -1 and price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0
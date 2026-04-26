#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1dRegime_v1
Hypothesis: Ichimoku cloud (TK cross + cloud filter) on 6h with 1d trend regime (ADX>25) to capture strong trends while avoiding chop. Uses volume confirmation (>1.5x median) for entry quality. Designed for 6h timeframe to target 12-30 trades/year. Works in bull/bear by requiring strong trend alignment and filtering false breaks in low-volatility regimes.
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
    
    # Get 1d data for HTF trend regime (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend regime filter
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = np.zeros_like(dx)
    adx_1d[27] = np.mean(dx[14:28])  # First ADX after 2*period-1
    for i in range(28, len(dx)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Trend regime: ADX > 25 = trending market
    trending_regime = adx_1d > 25
    
    # Align HTF regime to 6h
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime.astype(float))
    
    # Ichimoku components on 6h data
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
    
    # Current Ichimoku cloud (using Senkou A/B shifted back 26 periods to align with price)
    # For cloud twist/filter, we use current Senkou A/B (which are plotted 26 periods ahead)
    # So to get cloud at current price, we look at Senkou A/B that were calculated 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross (Tenkan/Kijun cross)
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume confirmation: volume > 1.5x median volume (50-period)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > 1.5 * vol_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku components (52), volume median (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trending_regime_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        trending = trending_regime_aligned[i] > 0.5
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        above_cloud = price_above_cloud[i]
        below_cloud = price_below_cloud[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + trending regime + volume confirmation
            long_signal = tk_up and above_cloud and trending and vol_conf
            # Short: TK cross down + price below cloud + trending regime + volume confirmation
            short_signal = tk_down and below_cloud and trending and vol_conf
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when price closes below cloud or TK cross down
            signals[i] = 0.25
            if close[i] < cloud_bottom[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when price closes above cloud or TK cross up
            signals[i] = -0.25
            if close[i] > cloud_top[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1dRegime_v1"
timeframe = "6h"
leverage = 1.0
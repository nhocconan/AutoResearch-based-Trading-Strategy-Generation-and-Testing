#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: On 6h timeframe, Ichimoku cloud (Tenkan/Kijun/Senkou) combined with 1d trend filter (price > 1d EMA50) captures strong momentum breaks while avoiding sideways chop. Long when price breaks above cloud with bullish TK cross; Short when price breaks below cloud with bearish TK cross. Uses discrete sizing (±0.25) and close-based exits. Designed for 12-30 trades/year on BTC/ETH with edge in both bull/bear regimes via trend alignment and cloud twist filter.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Cloud top/bottom (Senkou Span A/B shifted forward 26 periods)
    # For signal at bar i, we use Senkou values that were known 26 periods ago
    # So we shift senkou_a/b BACK by 26 to align with current price
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top is max of Senkou A/B, cloud bottom is min
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku periods (52) + alignment buffer
    start_idx = 52 + 26  # 52 for calculation, 26 for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from rolling or alignment)
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(tk_bullish[i]) or np.isnan(tk_bearish[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        tk_bull = tk_bullish[i]
        tk_bear = tk_bearish[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Bullish breakout: price above cloud AND bullish TK cross AND price > 1d EMA50
        long_entry = (close_val > cloud_top_val) and tk_bull and (close_val > ema_50_val)
        
        # Bearish breakout: price below cloud AND bearish TK cross AND price < 1d EMA50
        short_entry = (close_val < cloud_bottom_val) and tk_bear and (close_val < ema_50_val)
        
        # Exit: price re-enters cloud OR TK cross reverses OR price crosses 1d EMA50 against trend
        long_exit = (close_val < cloud_top_val) or (not tk_bull) or (close_val < ema_50_val)
        short_exit = (close_val > cloud_bottom_val) or (not tk_bear) or (close_val > ema_50_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0
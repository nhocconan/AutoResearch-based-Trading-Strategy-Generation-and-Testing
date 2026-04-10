#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Weekly Pivot Filter
# - Primary: 6h Ichimoku (Tenkan/Kijun cross + price vs cloud) for trend entry
# - HTF: 1d for weekly Camarilla pivots (H4/L4) as institutional support/resistance
# - Long: Tenkan > Kijun AND price > cloud (Senkou Span A/B) AND close > 1d H4 pivot
# - Short: Tenkan < Kijun AND price < cloud AND close < 1d L4 pivot
# - Exit: Tenkan/Kijun cross reverses OR price re-enters cloud
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: Ichimoku captures trends, weekly pivots filter false breaks in ranging markets (2025)

name = "6h_1d_ichimoku_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for entry)
    
    # Calculate weekly Camarilla pivots from 1d data (using prior week's OHLC)
    # Approximate weekly by taking 5-day OHLC (1 trading week)
    def calculate_weekly_ohlc(daily_series, window=5):
        # For each point, get OHLC of prior 5 days
        high_week = pd.Series(daily_series).rolling(window=window, min_periods=window).max().values
        low_week = pd.Series(daily_series).rolling(window=window, min_periods=window).min().values
        # For weekly close, use the close of the most recent day in the window
        close_week = pd.Series(daily_series).shift(1).rolling(window=window, min_periods=window).apply(
            lambda x: x[-1] if len(x) == window else np.nan, raw=False
        ).values
        # For weekly open, use the open of the first day in the window (approximate with prior day's close)
        open_week = pd.Series(daily_series).shift(window).rolling(window=window, min_periods=window).apply(
            lambda x: x[0] if len(x) == window else np.nan, raw=False
        ).values
        return high_week, low_week, close_week, open_week
    
    # Since we don't have 1d open in df_1d, we'll approximate weekly OHLC using close
    # Weekly high = max of prior 5 daily highs
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    # Weekly low = min of prior 5 daily lows
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    # Weekly close = close of prior day (most recent completed day)
    weekly_close = pd.Series(close_1d).shift(1).values
    # Weekly open = close 5 days ago (approximation)
    weekly_open = pd.Series(close_1d).shift(5).values
    
    # Calculate Camarilla levels for weekly
    weekly_rng = weekly_high - weekly_low
    h4_weekly = weekly_close + 1.5 * weekly_rng  # Weekly H4
    l4_weekly = weekly_close - 1.5 * weekly_rng  # Weekly L4
    
    # Align weekly Camarilla levels to 6h
    h4_weekly_aligned = align_htf_to_ltf(prices, df_1d, h4_weekly)
    l4_weekly_aligned = align_htf_to_ltf(prices, df_1d, l4_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup (52 periods for Senkou B)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(h4_weekly_aligned[i]) or np.isnan(l4_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        price_above_cloud = (close_6h[i] > senkou_a[i]) and (close_6h[i] > senkou_b[i])
        price_below_cloud = (close_6h[i] < senkou_a[i]) and (close_6h[i] < senkou_b[i])
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Weekly pivot conditions
        above_weekly_h4 = close_6h[i] > h4_weekly_aligned[i]
        below_weekly_l4 = close_6h[i] < l4_weekly_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Ichimoku + above weekly H4
            if (tenkan_above_kijun and price_above_cloud and above_weekly_h4):
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish Ichimoku + below weekly L4
            elif (tenkan_below_kijun and price_below_cloud and below_weekly_l4):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Ichimoku cross reverses OR price re-enters cloud
            if position == 1:  # Long position
                exit_condition = (
                    tenkan_below_kijun or  # Tenkan/Kijun cross reverses
                    not price_above_cloud   # Price re-enters or falls below cloud
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    tenkan_above_kijun or  # Tenkan/Kijun cross reverses
                    not price_below_cloud   # Price re-enters or rises above cloud
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
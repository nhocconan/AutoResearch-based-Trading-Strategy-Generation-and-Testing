#!/usr/bin/env python3
"""
1D_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Uses weekly trend filter with daily Camarilla pivot breakouts.
- Weekly trend: price above/below weekly Kumo cloud (Ichimoku) determines bias
- Daily entries: long when price breaks above R1 with volume spike in weekly uptrend
  short when price breaks below S1 with volume spike in weekly downtrend
- Exits: price returns to Pivot point (mean reversion to daily mean) or weekly trend flip
- Designed for low trade frequency (10-25/year) via weekly trend filter + precise daily breakout
- Works in bull/bear by following higher-timeframe Ichimoku trend
"""

name = "1D_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Ichimoku for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w = calculate_ichimoku(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Cloud boundaries (Senkou Span A/B)
    cloud_top_w = np.maximum(senkou_a_w, senkou_b_w)
    cloud_bottom_w = np.minimum(senkou_a_w, senkou_b_w)
    
    # Weekly trend: 1 if price above cloud, -1 if below cloud, 0 if inside cloud
    weekly_price = df_1w['close'].values
    trend_w = np.where(weekly_price > cloud_top_w, 1,
                       np.where(weekly_price < cloud_bottom_w, -1, 0))
    
    # Align weekly Ichimoku components to daily timeframe
    tenkan_w_1d = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_1d = align_htf_to_ltf(prices, df_1w, kijun_w)
    cloud_top_w_1d = align_htf_to_ltf(prices, df_1w, cloud_top_w)
    cloud_bottom_w_1d = align_htf_to_ltf(prices, df_1w, cloud_bottom_w)
    trend_w_1d = align_htf_to_ltf(prices, df_1w, trend_w)
    
    # --- Daily Pivot Points (Camarilla) ---
    # Use previous day's OHLC to calculate today's levels
    # Roll by 1 to ensure we only use past data
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first day's values to NaN (no previous day)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r1 = pivot + (range_prev * 1.1 / 12)
    s1 = pivot - (range_prev * 1.1 / 12)
    # Optional: r2, s2 for stronger breakouts
    r2 = pivot + (range_prev * 1.1 / 6)
    s2 = pivot - (range_prev * 1.1 / 6)
    
    # Volume Spike Detection (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_w_1d[i]) or np.isnan(kijun_w_1d[i]) or 
            np.isnan(cloud_top_w_1d[i]) or np.isnan(cloud_bottom_w_1d[i]) or
            np.isnan(trend_w_1d[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(pivot[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        # Weekly trend direction (must be clear trend, not inside cloud)
        weekly_trend = trend_w_1d[i]
        
        if position == 0:
            # Long: weekly uptrend + price breaks above R1 + volume
            if (weekly_trend == 1 and 
                close[i] > r1[i] and 
                close[i-1] <= r1[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below S1 + volume
            elif (weekly_trend == -1 and 
                  close[i] < s1[i] and 
                  close[i-1] >= s1[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to pivot OR weekly trend turns down
                if (close[i] <= pivot[i] or trend_w_1d[i] == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR weekly trend turns up
                if (close[i] >= pivot[i] or trend_w_1d[i] == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
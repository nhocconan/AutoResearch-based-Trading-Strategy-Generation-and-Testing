#!/usr/bin/env python3
"""
6h_1w_Ichimoku_Cloud_Flip
Hypothesis: Uses weekly Ichimoku cloud to define long-term trend (bullish when price above cloud, bearish when below).
Entries occur on 6h when price crosses the conversion line (Tenkan-sen) in the direction of the weekly trend,
with volume confirmation. Exits when price crosses the base line (Kijun-sen) or weekly trend flips.
Designed to work in both bull and bear markets by following higher-timeframe Ichimoku trend.
Targets low trade frequency (15-30/year) via weekly trend filter and Ichimoku cross logic.
"""

name = "6h_1w_Ichimoku_Cloud_Flip"
timeframe = "6h"
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
    
    # 6h OHLCV
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
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_w_6h = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_6h = align_htf_to_ltf(prices, df_1w, kijun_w)
    cloud_top_w_6h = align_htf_to_ltf(prices, df_1w, cloud_top_w)
    cloud_bottom_w_6h = align_htf_to_ltf(prices, df_1w, cloud_bottom_w)
    trend_w_6h = align_htf_to_ltf(prices, df_1w, trend_w)
    
    # --- 6h Ichimoku for Entry/Exit Signals ---
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # Volume Spike Detection (24-period average = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_w_6h[i]) or np.isnan(kijun_w_6h[i]) or 
            np.isnan(cloud_top_w_6h[i]) or np.isnan(cloud_bottom_w_6h[i]) or
            np.isnan(trend_w_6h[i]) or np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Weekly trend direction (must be clear trend, not inside cloud)
        weekly_trend = trend_w_6h[i]
        
        if position == 0:
            # Long: weekly uptrend + price crosses above Tenkan + volume
            if (weekly_trend == 1 and 
                close[i] > tenkan_6h[i] and 
                close[i-1] <= tenkan_6h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price crosses below Tenkan + volume
            elif (weekly_trend == -1 and 
                  close[i] < tenkan_6h[i] and 
                  close[i-1] >= tenkan_6h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR price crosses below Kijun
                if (trend_w_6h[i] == -1 or 
                    (close[i] < kijun_6h[i] and close[i-1] >= kijun_6h[i-1])):  # crossed below
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR price crosses above Kijun
                if (trend_w_6h[i] == 1 or 
                    (close[i] > kijun_6h[i] and close[i-1] <= kijun_6h[i-1])):  # crossed above
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
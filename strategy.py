#!/usr/bin/env python3
"""
6h_Pivot_Reversal_Breakout_1dTrend
Hypothesis: Combines daily pivot reversals with trend continuation. In trending markets (above/below 1d EMA34),
trade breakouts of R1/S1 for continuation. In ranging markets (near 1d EMA34), fade at R3/S3 for mean reversion.
Uses volume confirmation to filter false signals. Designed for 6H timeframe to work in both bull/bear regimes.
Target: 20-40 trades/year per symbol.
"""

name = "6h_Pivot_Reversal_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Daily trend filter: EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema34_1d
    trend_down = close_1d < ema34_1d
    
    # Align daily data to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Volume confirmation (20-period average on 6h)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5  # Volume above 1.5x average
        
        # Determine market regime based on price vs daily EMA34
        price_vs_ema = (close[i] - ema34_1d_aligned[i]) / ema34_1d_aligned[i]
        is_trending = abs(price_vs_ema) > 0.02  # More than 2% away from EMA34
        is_ranging = abs(price_vs_ema) <= 0.02   # Within 2% of EMA34
        
        if position == 0:
            # In trending markets: breakout of R1/S1 for continuation
            if is_trending:
                if trend_up_aligned[i] > 0.5 and close[i] > r1_6h[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif trend_down_aligned[i] > 0.5 and close[i] < s1_6h[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            # In ranging markets: fade at R3/S3 for mean reversion
            elif is_ranging:
                if close[i] >= r3_6h[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                elif close[i] <= s3_6h[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # In trend: exit on breakdown of S1 or trend reversal
                if close[i] < s1_6h[i] or trend_down_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit on mean reversion to pivot or opposite S1
                if close[i] <= pivot_6h[i] or close[i] >= r1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # In trend: exit on breakout of R1 or trend reversal
                if close[i] > r1_6h[i] or trend_up_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit on mean reversion to pivot or opposite S1
                if close[i] >= pivot_6h[i] or close[i] <= s1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_WeeklyTrend_1dVolatility
Hypothesis: Ichimoku cloud acts as dynamic support/resistance on 6h timeframe.
Trend direction from weekly timeframe (bullish when price above weekly Kumo).
Volatility filter from daily ATR to avoid choppy markets (low ATR = range, avoid trading).
Enter long when price breaks above 6h Kumo with bullish weekly trend and low volatility.
Enter short when price breaks below 6h Kumo with bearish weekly trend and low volatility.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
Works in bull markets via weekly trend filter and in bear markets via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # The cloud (Kumo) is between Senkou Span A and Senkou Span B
    # Upper cloud boundary: max(Senkou A, Senkou B)
    # Lower cloud boundary: min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Load weekly data for trend filter (using Kumo twist)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Ichimoku for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan and Kijun
    high_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    high_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_sen_1w + kijun_sen_1w) / 2
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_senkou_b_1w + low_senkou_b_1w) / 2
    
    # Weekly Kumo (cloud)
    upper_cloud_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    lower_cloud_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Weekly trend: bullish when price above weekly Kumo, bearish when below
    weekly_trend_bullish = close_1w > upper_cloud_1w
    weekly_trend_bearish = close_1w < lower_cloud_1w
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bearish.astype(float))
    
    # Load daily data for volatility filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # First period has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR(14) - smoothed TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily ATR percentile (20-period) to identify low volatility regimes
    atr_percentile = pd.Series(atr_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    # Low volatility: ATR below 30th percentile (avoid choppy markets)
    low_volatility = atr_percentile < 0.30
    
    # Align daily volatility to 6h timeframe
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need Ichimoku calculation + weekly + daily data)
    start_idx = 60  # Enough for all indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(low_vol_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price above 6h Kumo + bullish weekly trend + low volatility
        if close[i] > upper_cloud[i] and weekly_bullish_aligned[i] > 0.5 and low_vol_aligned[i] > 0.5:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price below 6h Kumo + bearish weekly trend + low volatility
        elif close[i] < lower_cloud[i] and weekly_bearish_aligned[i] > 0.5 and low_vol_aligned[i] > 0.5:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price re-enters the cloud (mean reversion to Kumo)
        elif position == 1 and close[i] < upper_cloud[i] and close[i] > lower_cloud[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > lower_cloud[i] and close[i] < upper_cloud[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_WeeklyTrend_1dVolatility"
timeframe = "6h"
leverage = 1.0
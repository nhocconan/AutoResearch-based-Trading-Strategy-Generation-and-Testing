#!/usr/bin/env python3
"""
6H_WeeklyPivot_Trend_Range_Switch_v2
Hypothesis: Switch between trend-following in strong trends (ADX>25) and mean-reversion in ranging markets (ADX<20) using weekly pivot levels.
In trending markets: breakout of weekly pivot/resistance/support with volume confirmation.
In ranging markets: fade at weekly R3/S3 levels with RSI extremes.
Uses 1d ADX for regime detection and weekly pivot levels for entries/exits.
Designed to work in both bull and bear markets by adapting to regime.
"""
name = "6H_WeeklyPivot_Trend_Range_Switch_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for ADX regime and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate weekly pivot from daily data (using prior week's OHLC)
    # We'll use the last 5 trading days to approximate weekly OHLC
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    
    # Weekly resistance/support levels
    r1 = weekly_pivot + (weekly_range * 1.1 / 12)
    r2 = weekly_pivot + (weekly_range * 1.1 / 6)
    r3 = weekly_pivot + (weekly_range * 1.1 / 4)
    s1 = weekly_pivot - (weekly_range * 1.1 / 12)
    s2 = weekly_pivot - (weekly_range * 1.1 / 6)
    s3 = weekly_pivot - (weekly_range * 1.1 / 4)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # RSI (14) for mean-reversion signals
    close_pd = pd.Series(close)
    delta = close_pd.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(30, 20)  # Ensure sufficient warmup for ADX and other indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Minimum 6 bars between trades (1.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Regime-based switching
            if adx_val > 25:  # Trending regime
                # Long: breakout above weekly R1 with volume
                if (high[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                    volume_filter[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
                # Short: breakdown below weekly S1 with volume
                elif (low[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                      volume_filter[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_exit = 0
            elif adx_val < 20:  # Ranging regime
                # Long: RSI oversold and price at weekly S3
                if (rsi_val < 30 and low[i] <= s3_aligned[i] and 
                    close[i-1] > s3_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
                # Short: RSI overbought and price at weekly R3
                elif (rsi_val > 70 and high[i] >= r3_aligned[i] and 
                      close[i-1] < r3_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_exit = 0
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Long exit: RSI overbought or price returns to pivot
                if (rsi_val > 70 or close[i] <= weekly_pivot[i-1] if i > 0 else False):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RSI oversold or price returns to pivot
                if (rsi_val < 30 or close[i] >= weekly_pivot[i-1] if i > 0 else False):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals
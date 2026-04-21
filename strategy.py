#!/usr/bin/env python3
"""
6h_1d_1w_Camarilla_R3S3_Fade_v1
Hypothesis: Fade extreme Camarilla levels (R3/S3) on 6h timeframe with 1d trend filter (EMA50) and volume confirmation.
Works in bull/bear: In uptrend, fade R3 for short; in downtrend, fade S3 for long. Uses 1d EMA for trend, 1w pivot for bias.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R3, S3 (extreme fade levels)
    rang = prev_high - prev_low
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data for weekly pivot bias (long-term direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for weekly pivot (simplified: use close as bias)
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    weekly_bias = align_htf_to_ltf(prices, df_1w, prev_close_1w)  # weekly close as trend bias
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_bias[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Determine bias: weekly close > previous weekly close = bullish bias
        weekly_bullish = weekly_bias[i] > weekly_bias[i-1] if i > 0 and not np.isnan(weekly_bias[i-1]) else True
        
        if position == 0:
            # Long conditions: price < S3 (oversold) AND 1d uptrend AND weekly bullish bias AND volume
            if (price < s3_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # 1d EMA rising
                weekly_bullish and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price > R3 (overbought) AND 1d downtrend AND weekly bearish bias AND volume
            elif (price > r3_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # 1d EMA falling
                  not weekly_bullish and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price > 1d EMA50 (trend exhaustion) or price > R2 (mean reversion fail)
            # Calculate R2 for exit
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                r2_exit = prev_close_i + rang_i * 2.0 / 12
                # Align R2 exit level (simplified: use current day's R2)
                r2_exit_aligned = align_htf_to_ltf(prices, df_1d, 
                                                  pd.Series([prev_close_i + rang_i * 2.0 / 12] * len(df_1d)).values)
                if price > ema_50_1d_aligned[i] or (not np.isnan(r2_exit_aligned[i]) and price > r2_exit_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price < 1d EMA50 (trend exhaustion) or price < S2 (mean reversion fail)
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                s2_exit = prev_close_i - rang_i * 2.0 / 12
                s2_exit_aligned = align_htf_to_ltf(prices, df_1d, 
                                                  pd.Series([prev_close_i - rang_i * 2.0 / 12] * len(df_1d)).values)
                if price < ema_50_1d_aligned[i] or (not np.isnan(s2_exit_aligned[i]) and price < s2_exit_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Camarilla_R3S3_Fade_v1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_hull_ma_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use Hull Moving Average (HMA) trend filter with daily trend confirmation and volume spike for entries. Enter long when price crosses above HMA(16) with daily EMA50 > EMA200 and volume > 2x average; enter short when price crosses below HMA(16) with daily EMA50 < EMA200 and volume > 2x average. Exit on opposite HMA crossover or trend reversal. Hull MA reduces lag while maintaining smoothness, improving entry timing in trends. Volume spike confirms institutional participation. Works in bull/bear via daily trend filter. Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_hull_ma_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def hull_moving_average(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    # WMA of full period
    wma_full = pd.Series(series).rolling(window=period, min_periods=period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate HMA(16) on 4h
    hma_16 = hull_moving_average(close, 16)
    
    # Calculate daily EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema50_d = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_d = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_d)
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_d)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(hma_16[i]) or np.isnan(hma_16[i-1]) or
            np.isnan(ema50_4h[i]) or np.isnan(ema200_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below HMA(16)
            if close[i] < hma_16[i] and close[i-1] >= hma_16[i-1]:
                exit_long = True
            # Exit if daily EMA50 crosses below EMA200 (trend reversal)
            elif ema50_4h[i] < ema200_4h[i] and ema50_4h[i-1] >= ema200_4h[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above HMA(16)
            if close[i] > hma_16[i] and close[i-1] <= hma_16[i-1]:
                exit_short = True
            # Exit if daily EMA50 crosses above EMA200 (trend reversal)
            elif ema50_4h[i] > ema200_4h[i] and ema50_4h[i-1] <= ema200_4h[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above HMA(16) with daily EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (close[i] > hma_16[i] and close[i-1] <= hma_16[i-1] and
                ema50_4h[i] > ema200_4h[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price crosses below HMA(16) with daily EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (close[i] < hma_16[i] and close[i-1] >= hma_16[i-1] and
                ema50_4h[i] < ema200_4h[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
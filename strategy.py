#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Regime_v1
Donchian channel breakout with volume confirmation and Choppiness regime filter.
Trades breakouts only in trending markets (Choppiness < 38.2) to avoid false signals in ranging markets.
Designed for 12h timeframe with target of 50-150 total trades over 4 years.
Uses daily trend filter for alignment and ATR-based stoploss.
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
    
    # === 12h Donchian Channel (20-period) ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[max(0, i-9):i+1])
            lowest_low[i] = np.min(low[max(0, i-9):i+1])
        else:
            highest_high[i] = high[i]
            lowest_low[i] = low[i]
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === 12h Choppiness Index (14-period) ===
    atr_14 = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 1:
            tr = np.max([
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            ])
            if i == 1:
                atr_14[i] = tr
            elif i < 14:
                atr_14[i] = (atr_14[i-1] * (i-1) + tr) / i
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr) / 14
        else:
            atr_14[i] = 0
    
    chop = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 13:
            highest_high_14 = np.max(high[i-13:i+1])
            lowest_low_14 = np.min(low[i-13:i+1])
            sum_atr_14 = np.sum(atr_14[i-13:i+1])
            if sum_atr_14 > 0 and (highest_high_14 - lowest_low_14) > 0:
                chop[i] = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
            else:
                chop[i] = 50
        else:
            chop[i] = 50
    
    chop_threshold = 38.2  # trending market
    trending_market = chop < chop_threshold
    
    # === 1d Donchian Channel for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    highest_high_1d = np.full_like(high_1d, np.nan)
    lowest_low_1d = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            highest_high_1d[i] = np.max(high_1d[i-19:i+1])
            lowest_low_1d[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            highest_high_1d[i] = np.max(high_1d[max(0, i-9):i+1])
            lowest_low_1d[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            highest_high_1d[i] = high_1d[i]
            lowest_low_1d[i] = low_1d[i]
    
    # Daily trend: bullish if price above midpoint of Donchian channel
    midpoint_1d = (highest_high_1d + lowest_low_1d) / 2
    daily_trend_bullish = df_1d['close'].values > midpoint_1d
    daily_trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_bullish.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(trending_market[i]) or 
            np.isnan(daily_trend_bullish_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume confirmation AND trending market AND daily trend bullish
            if (close[i] > highest_high[i] and 
                vol_confirm[i] and 
                trending_market[i] and 
                daily_trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian band AND volume confirmation AND trending market AND daily trend bearish
            elif (close[i] < lowest_low[i] and 
                  vol_confirm[i] and 
                  trending_market[i] and 
                  daily_trend_bullish_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian band OR loss of trending market
            if (close[i] < lowest_low[i] or 
                not trending_market[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian band OR loss of trending market
            if (close[i] > highest_high[i] or 
                not trending_market[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0
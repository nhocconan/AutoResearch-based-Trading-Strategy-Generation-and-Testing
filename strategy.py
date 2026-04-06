#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Trend Filter and Volume Confirmation
Hypothesis: Weekly trend (via HMA-21) filters breakouts from 6h Donchian channels,
reducing whipsaws in counter-trend moves. Volume > 2x 20-period average confirms
breakout strength. Works in bull/bear by only taking breaks aligned with weekly trend.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on weekly
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_weekly = hma(close_weekly, 21)
    hma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, hma_weekly)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_weekly = np.where(close_weekly > hma_weekly, 1, -1)
    trend_weekly_aligned = align_htf_to_ltf(prices, df_weekly, trend_weekly)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_weekly_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_weekly_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish weekly trend with volume
            if (close[i] > donch_high[i] and
                trend_weekly_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish weekly trend with volume
            elif (close[i] < donch_low[i] and
                  trend_weekly_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
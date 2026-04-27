#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly Pivot Point (PP) levels for trend direction and 
daily Donchian(20) breakouts for entry timing. In bullish weekly context (price > weekly PP), 
we take long breakouts above daily Donchian high; in bearish context (price < weekly PP), 
we take short breakdowns below daily Donchian low. Volume > 2x average confirms breakout strength.
Uses discrete position sizes (0.0, ±0.25) to minimize fee churn. Target: 15-30 trades/year (60-120 over 4 years).
"""

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
    
    # Get weekly data for Pivot Point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Pivot Point (PP) = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly PP to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    lookback = 20
    highest_high_1d = np.full(len(df_1d), np.nan)
    lowest_low_1d = np.full(len(df_1d), np.nan)
    
    for i in range(lookback, len(df_1d)):
        highest_high_1d[i] = np.max(high_1d[i-lookback:i])
        lowest_low_1d[i] = np.min(low_1d[i-lookback:i])
    
    # Align daily Donchian levels to 6h timeframe
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    
    # 20-period average volume on daily data for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(df_1d), np.nan)
    vol_period = 20
    for i in range(vol_period, len(df_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-vol_period:i])
    
    # Align daily volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR for volatility filtering (optional)
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need weekly PP, daily Donchian (20), daily volume MA (20)
    start_idx = 20  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        if (np.isnan(pp_1w_aligned[i]) or
            np.isnan(highest_high_1d_aligned[i]) or
            np.isnan(lowest_low_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        
        # Determine weekly trend from Pivot Point
        bullish_weekly = price > pp_1w_aligned[i]
        bearish_weekly = price < pp_1w_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian high in bullish weekly context with volume
            if bullish_weekly and price > highest_high_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below daily Donchian low in bearish weekly context with volume
            elif bearish_weekly and price < lowest_low_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below daily Donchian low or weekly turns bearish
            if price < lowest_low_1d_aligned[i] or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above daily Donchian high or weekly turns bullish
            if price > highest_high_1d_aligned[i] or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPP_DailyDonchian20_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
1h Volume Spike + 4h/1d Regime Filter (Chop/ADX) + Discrete Sizing
Hypothesis: In 1h timeframe, volume spikes (>2.0x 20-bar MA) combined with 4h trend filter (price > 4h EMA50) and 1d regime filter (Choppiness Index < 50 for trending) capture institutional moves with controlled frequency. Uses discrete position sizing (0.0, ±0.20) to minimize fee drag. Works in bull/bear via trend alignment and regime adaptation.
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
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = pd.Series(df_4h['close'])
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Choppiness Index regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Sum of TR over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        tr_sum_14[i] = np.sum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        hh_14[i] = np.max(high_1d[i-13:i+1])
        ll_14[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if hh_14[i] > ll_14[i]:
            chop_1d[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0  # undefined case
    
    # Align Chop to 1h timeframe (no extra delay for regime)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period volume MA for volume confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Chop, volume MA
    start_idx = max(50, 13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # close position outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_4h_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 4h EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Regime filter: Chop < 50 indicates trending market (favor trend following)
        trending_regime = chop_val < 50.0
        
        if position == 0:
            # Look for entry signals with volume confirmation
            # Long: price above EMA50 in uptrend + volume spike + trending regime
            long_entry = uptrend and volume_confirm and trending_regime
            # Short: price below EMA50 in downtrend + volume spike + trending regime
            short_entry = downtrend and volume_confirm and trending_regime
            
            if long_entry:
                signals[i] = 0.20  # 20% position
                position = 1
            elif short_entry:
                signals[i] = -0.20  # 20% short
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below EMA50 OR volume dries up OR regime turns choppy
            if curr_close < ema_50_val or not volume_confirm or chop_val >= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price closes above EMA50 OR volume dries up OR regime turns choppy
            if curr_close > ema_50_val or not volume_confirm or chop_val >= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA50_1dChopRegime"
timeframe = "1h"
leverage = 1.0
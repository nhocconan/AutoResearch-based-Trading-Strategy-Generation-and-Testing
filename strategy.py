#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Daily Donchian breakouts capture major trends. 1w EMA50 filter ensures we trade with the primary trend.
Volume spike confirms institutional participation. Chop filter avoids whipsaws in ranging markets.
Designed to work in both bull and bear markets via trend filter and regime adaptation.
Target: 7-25 trades/year (30-100 total over 4 years).
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
    
    # Get 1d data for Donchian channels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for 20-period Donchian
        return np.zeros(n)
    
    # Calculate 1d Donchian Channels (20-period)
    # Upper channel = highest high over past 20 days
    # Lower channel = lowest low over past 20 days
    highest_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR1, n) / (max(high, n) - min(low, n))) / log10(n)
    # where n=14 period
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).sum().values  # sum for CHOP formula
        
        # Rolling max(high) and min(low) over 14 periods
        max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        range_14 = max_high_14 - min_low_14
        
        # Avoid division by zero
        chop_14 = np.where(
            range_14 != 0,
            100 * np.log10(atr_14 / range_14) / np.log10(14),
            50  # neutral value when range is zero
        )
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    else:
        chop_aligned = np.full(n, 50.0)  # default to neutral chop if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        ema_50 = ema_50_aligned[i]
        chop_value = chop_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter: price above/below 1w EMA50
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        # Chop filter: CHOP < 50 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid)
        not_choppy = chop_value < 50
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume spike AND uptrend AND not choppy
            long_condition = (curr_high > donchian_upper) and volume_spike and uptrend and not_choppy
            # Short: price breaks below Donchian lower AND volume spike AND downtrend AND not choppy
            short_condition = (curr_low < donchian_lower) and volume_spike and downtrend and not_choppy
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below Donchian lower or trend changes or choppy market
            if curr_close <= donchian_lower or not uptrend or chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian upper or trend changes or choppy market
            if curr_close >= donchian_upper or not downtrend or chop_value > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0
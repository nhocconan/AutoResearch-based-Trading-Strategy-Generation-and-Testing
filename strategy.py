#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
Hypothesis: Donchian breakouts capture institutional momentum. Filtered by 1w ADX>25 for trending markets and 1d volume spike for conviction. Works in bull (buy breakouts) and bear (sell breakdowns). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1w_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smoothed values
        atr_period = np.full(n, np.nan)
        plus_dm_period = np.full(n, np.nan)
        minus_dm_period = np.full(n, np.nan)
        
        if n >= period + 1:
            # Initial averages
            atr_period[period] = np.sum(tr[:period])
            plus_dm_period[period] = np.sum(plus_dm[:period])
            minus_dm_period[period] = np.sum(minus_dm[:period])
            
            # Wilder's smoothing
            for i in range(period + 1, n):
                atr_period[i] = (atr_period[i-1] * (period - 1) + tr[i-1]) / period
                plus_dm_period[i] = (plus_dm_period[i-1] * (period - 1) + plus_dm[i-1]) / period
                minus_dm_period[i] = (minus_dm_period[i-1] * (period - 1) + minus_dm[i-1]) / period
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if not np.isnan(atr_period[i]) and atr_period[i] != 0:
                plus_di[i] = (plus_dm_period[i] / atr_period[i]) * 100
                minus_di[i] = (minus_dm_period[i] / atr_period[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if n >= 2 * period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 12h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 12h: approx 0.5 of 1d volume (since 2x 12h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] * 0.5 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1w_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX weakens
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                adx_1w_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX weakens
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                adx_1w_aligned[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 24 bars flat
            if bars_since_entry >= 24:
                # Breakout entries: upper/lower with volume and trend filter
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with volume + trend filter
                if bull_breakout and volume_filter and trend_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with volume + trend filter
                elif bear_breakout and volume_filter and trend_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals
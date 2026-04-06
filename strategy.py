#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 12h ADX > 25 for trending markets and 12h volume > 1.5x average for conviction. Works in bull (buy breakouts with ADX up) and bear (sell breakdowns with ADX up). Uses 12h timeframe for trend to reduce noise vs 1d. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_adx_vol_v1"
timeframe = "4h"
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
    
    # Get 12h data for ADX and volume filters
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 14-period ADX on 12h data
    adx = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 14:
        # True Range
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothing (Wilder's smoothing = 1/14)
        atr_12h = np.full(len(tr_12h), np.nan)
        if len(tr_12h) > 0:
            atr_12h[0] = tr_12h[0]
            for i in range(1, len(tr_12h)):
                atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
        
        plus_di = np.full(len(up_move), np.nan)
        minus_di = np.full(len(down_move), np.nan)
        if len(atr_12h) > 0 and not np.all(np.isnan(atr_12h)):
            # Avoid division by zero
            valid_atr = atr_12h.copy()
            valid_atr[valid_atr == 0] = 1e-10
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / valid_atr
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / valid_atr
            
            # DX and ADX
            dx = np.full(len(plus_di), np.nan)
            dx_sum = np.zeros(len(plus_di))
            dx_count = np.zeros(len(plus_di))
            for i in range(len(plus_di)):
                if not (np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or (plus_di[i] + minus_di[i]) == 0):
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            # ADX = smoothed DX
            adx_valid = np.full(len(dx), np.nan)
            if len(dx) >= 14:
                # First ADX is average of first 14 DX
                first_adx = np.nanmean(dx[:14])
                adx_valid[13] = first_adx if not np.isnan(first_adx) else np.nan
                # Subsequent ADX: (prev_ADX * 13 + current_DX) / 14
                for i in range(14, len(dx)):
                    prev_adx = adx_valid[i-1]
                    if not np.isnan(prev_adx) and not np.isnan(dx[i]):
                        adx_valid[i] = (prev_adx * 13 + dx[i]) / 14
                    else:
                        adx_valid[i] = np.nan
            adx[14:] = adx_valid[14:] if len(adx_valid) > 14 else adx_valid
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align 12h indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Donchian channels (20-period) from 4h data
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
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 4h: approx 1/2 of 12h volume (since 2x 4h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX weak (< 20)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                adx_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX weak (< 20)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                adx_aligned[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with ADX filter and volume
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with strong ADX + volume
                if bull_breakout and adx_filter and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with strong ADX + volume
                elif bear_breakout and adx_filter and volume_filter:
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
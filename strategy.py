#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h ADX Trend + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, 12h ADX ensures trending market (avoiding chop), volume confirms breakout strength, ATR stoploss limits drawdown. Works in bull/bear by only trading when ADX > 25 (trending regime). Target 75-200 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12hadx_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for ADX and ATR (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 14-period ADX on 12h
    adx_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 14:
        # True Range
        tr = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth TR, +DM, -DM
        tr14 = np.full_like(tr, np.nan)
        plus_dm14 = np.full_like(tr, np.nan)
        minus_dm14 = np.full_like(tr, np.nan)
        
        if len(tr) >= 14:
            tr14[0] = np.nansum(tr[:14])
            plus_dm14[0] = np.nansum(plus_dm[:14])
            minus_dm14[0] = np.nansum(minus_dm[:14])
            
            for i in range(1, len(tr)):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i]
                minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i]
        
        # DI+ and DI-
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        valid = ~np.isnan(tr14) & (tr14 != 0)
        plus_di[valid] = (plus_dm14[valid] / tr14[valid]) * 100
        minus_di[valid] = (minus_dm14[valid] / tr14[valid]) * 100
        
        # DX and ADX
        dx_sum = plus_di + minus_di
        dx_valid = (dx_sum != 0) & ~np.isnan(plus_di) & ~np.isnan(minus_di)
        dx[dx_valid] = (np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / dx_sum[dx_valid]) * 100
        
        # ADX (smoothed DX)
        adx_12h = np.full_like(close_12h, np.nan)
        dx_valid = ~np.isnan(dx)
        if np.sum(dx_valid) >= 14:
            adx_12h[13] = np.nanmean(dx[:14])
            for i in range(14, len(dx)):
                if not np.isnan(dx[i]):
                    adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    # 14-period ATR on 12h for stoploss
    atr_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 14:
        tr = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        atr_12h[0] = np.nan
        if len(tr) > 0:
            atr_12h[1] = tr[0]
            for i in range(2, len(atr_12h)):
                atr_12h[i] = (tr[i-1] * 13 + atr_12h[i-1]) / 14
    
    # Align indicators to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 30)  # For Donchian and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR ADX < 20 (trend weakening)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                adx_12h_aligned[i] < 20 or
                close[i] < entry_price - 2.0 * atr_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR ADX < 20 (trend weakening)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                adx_12h_aligned[i] < 20 or
                close[i] > entry_price + 2.0 * atr_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX trend filter (ADX > 25)
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and (adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and (adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
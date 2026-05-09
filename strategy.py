#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
Designed for low trade frequency (target: 20-40 trades/year) with strong edge in both bull and bear markets.
Breakouts occur only in direction of 4h ADX trend, confirmed by volume surge.
Uses discrete position sizing (0.25) to minimize churn.
"""

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        atr = np.full_like(close, np.nan)
        plus_di = np.full_like(close, np.nan)
        minus_di = np.full_like(close, np.nan)
        
        if len(close) >= period:
            # Initial averages
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth = np.nansum(plus_dm[1:period+1])
            minus_dm_smooth = np.nansum(minus_dm[1:period+1])
            
            plus_di[period-1] = 100 * plus_dm_smooth / atr[period-1] if atr[period-1] != 0 else 0
            minus_di[period-1] = 100 * minus_dm_smooth / atr[period-1] if atr[period-1] != 0 else 0
            
            # Wilder smoothing
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm[i]
                minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm[i]
                plus_di[i] = 100 * plus_dm_smooth / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_smooth / atr[i] if atr[i] != 0 else 0
        
        # DX and ADX
        dx = np.full_like(close, np.nan)
        adx = np.full_like(close, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(plus_di) | np.isnan(minus_di) | (plus_di + minus_di == 0)] = np.nan
        
        if len(close) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(close)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume ratio (current vs 20-period average)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 2*14-1, 20)  # Donchian, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        bullish_breakout = close[i] > highest_high[i]
        bearish_breakout = close[i] < lowest_low[i]
        strong_trend = adx[i] > 25
        volume_surge = volume_ratio[i] > 2.0
        
        if position == 0:
            # Enter long: bullish breakout + strong trend + volume surge
            if bullish_breakout and strong_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish breakout + strong trend + volume surge
            elif bearish_breakout and strong_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend weakens
            if close[i] < lowest_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend weakens
            if close[i] > highest_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
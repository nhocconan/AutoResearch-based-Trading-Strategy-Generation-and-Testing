#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_Follow
Hypothesis: KAMA adapts to market noise, capturing strong trends while avoiding whipsaws in range.
Long when price > KAMA + volume expansion + 1w ADX > 25 (trend regime).
Short when price < KAMA + volume expansion + 1w ADX > 25.
Uses 12h timeframe for signal generation, 1d for KAMA trend, 1w for ADX regime filter.
Designed for low trade frequency (<30/year) to minimize fee drag in bear markets.
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
    
    # Get 1d data for KAMA trend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs fixing - will compute properly below
    
    # Proper ER calculation: |close - close[10]| / sum(|diff|) over 10 periods
    def calculate_erkama(close_vals, length=10):
        change = np.abs(np.diff(close_vals, prepend=close_vals[0]))
        dir = np.abs(np.subtract(close_vals, np.roll(close_vals, length)))
        vol = np.convolve(change, np.ones(length), 'same')  # Simple moving sum of change
        er = np.where(vol != 0, dir / vol, 0)
        sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
        return sc
    
    # Simpler approach: use Kaufman's actual formula
    lookback = 10
    diff = np.abs(np.subtract(close_1d, np.roll(close_1d, lookback)))
    abs_diff = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    
    # Rolling sum of absolute differences
    def rolling_sum(arr, window):
        return np.convolve(arr, np.ones(window), 'full')[window-1:-window+1]
    
    if len(abs_diff) >= lookback:
        sum_abs_diff = rolling_sum(abs_diff, lookback)
        # Pad beginning
        sum_abs_diff = np.concatenate([np.full(lookback-1, np.nan), sum_abs_diff])
        er = np.where(sum_abs_diff > 0, diff / sum_abs_diff, 0)
        sc = (er * 0.6 + 0.0645) ** 2  # sc = [ER*(fastest - slowest) + slowest]^2
        
        # Calculate KAMA
        kama = np.full_like(close_1d, np.nan)
        kama[lookback] = close_1d[lookback]  # Start with close
        for i in range(lookback + 1, len(close_1d)):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    else:
        kama = np.full_like(close_1d, np.nan)
    
    # Align KAMA to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.subtract(high, low)
        tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
        tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where(np.subtract(high, np.roll(high, 1)) > np.subtract(np.roll(low, 1), low),
                           np.maximum(np.subtract(high, np.roll(high, 1)), 0), 0)
        dm_minus = np.where(np.subtract(np.roll(low, 1), low) > np.subtract(high, np.roll(high, 1)),
                            np.maximum(np.subtract(np.roll(low, 1), low), 0), 0)
        
        # Smooth TR, DM+
        atr = np.zeros_like(tr)
        dmplus_smooth = np.zeros_like(dm_plus)
        dmminus_smooth = np.zeros_like(dm_minus)
        
        # First values (simple average)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            dmplus_smooth[period-1] = np.mean(dm_plus[:period])
            dmminus_smooth[period-1] = np.mean(dm_minus[:period])
            
            # Wilder smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dmplus_smooth[i] = (dmplus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dmminus_smooth[i] = (dmminus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        dimplus = np.where(atr != 0, 100 * dmplus_smooth / atr, 0)
        dimminus = np.where(atr != 0, 100 * dmminus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((dimplus + dimminus) != 0, 100 * np.abs(dimplus - dimminus) / (dimplus + dimminus), 0)
        adx = np.zeros_like(dx)
        
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma_20[:19] = np.nan  # Not enough data for first 19
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):  # Wait for sufficient data
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (trend up)
        # 2. Volume expansion
        # 3. ADX > 25 (trending market)
        long_condition = (close[i] > kama_aligned[i]) and volume_expansion[i] and (adx_1w_aligned[i] > 25)
        
        # Short conditions:
        # 1. Price below KAMA (trend down)
        # 2. Volume expansion
        # 3. ADX > 25 (trending market)
        short_condition = (close[i] < kama_aligned[i]) and volume_expansion[i] and (adx_1w_aligned[i] > 25)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_KAMA_Trend_Follow"
timeframe = "12h"
leverage = 1.0
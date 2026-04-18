#!/usr/bin/env python3
"""
Hypothesis: 1h mean reversion with 4h Bollinger Bands and 1d ADX trend filter.
In range-bound markets (ADX < 25): price reverts to Bollinger Band mean (20 SMA).
In trending markets (ADX >= 25): trade pullbacks to the 20 SMA in trend direction.
Volume confirms momentum. Designed for 15-30 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    sma = np.full(len(close), np.nan)
    if len(close) < period:
        return sma
    sma[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        sma[i] = (sma[i-1] * (period-1) + close[i]) / period
    return sma

def calculate_std(close, period):
    """Calculate Standard Deviation."""
    std = np.full(len(close), np.nan)
    if len(close) < period:
        return std
    for i in range(period-1, len(close)):
        std[i] = np.std(close[i-period+1:i+1])
    return std

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    if len(tr) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus = np.full(len(dm_plus), np.nan)
    di_minus = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
        else:
            di_plus[i] = 0
            di_minus[i] = 0
    
    # DX and ADX
    dx = np.full(len(di_plus), np.nan)
    for i in range(period, len(di_plus)):
        if (di_plus[i] + di_minus[i]) != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            dx[i] = 0
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands (20, 2)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands on 4h
    sma_20_4h = calculate_sma(close_4h, 20)
    std_20_4h = calculate_std(close_4h, 20)
    upper_bb_4h = sma_20_4h + 2 * std_20_4h
    lower_bb_4h = sma_20_4h - 2 * std_20_4h
    
    # Calculate ADX on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 1h timeframe
    upper_bb_4h_1h = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_1h = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    sma_20_4h_1h = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    adx_14_1d_1h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_4h_1h[i]) or np.isnan(lower_bb_4h_1h[i]) or 
            np.isnan(sma_20_4h_1h[i]) or np.isnan(adx_14_1d_1h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Determine market regime
            is_trending = adx_14_1d_1h[i] >= 25
            
            if is_trending:
                # Trending market: trade pullbacks to 20 SMA
                # Long: price near lower BB in uptrend (close > SMA)
                if close[i] <= lower_bb_4h_1h[i] * 1.02 and close[i] > sma_20_4h_1h[i] and vol_confirmed:
                    signals[i] = 0.20
                    position = 1
                # Short: price near upper BB in downtrend (close < SMA)
                elif close[i] >= upper_bb_4h_1h[i] * 0.98 and close[i] < sma_20_4h_1h[i] and vol_confirmed:
                    signals[i] = -0.20
                    position = -1
            else:
                # Ranging market: mean reversion at BB extremes
                # Long: price at or below lower BB
                if close[i] <= lower_bb_4h_1h[i] and vol_confirmed:
                    signals[i] = 0.20
                    position = 1
                # Short: price at or above upper BB
                elif close[i] >= upper_bb_4h_1h[i] and vol_confirmed:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses above 20 SMA or reaches upper BB
            if close[i] >= sma_20_4h_1h[i] or close[i] >= upper_bb_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses below 20 SMA or reaches lower BB
            if close[i] <= sma_20_4h_1h[i] or close[i] <= lower_bb_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BB20_4hADX_Volume"
timeframe = "1h"
leverage = 1.0
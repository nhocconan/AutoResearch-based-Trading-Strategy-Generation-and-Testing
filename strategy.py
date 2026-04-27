#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot levels (using last full week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from last 5 daily bars (weekly lookback)
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    for i in range(5, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # previous day's close
    
    # Weekly pivot levels (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Get 12h data for trend filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        dm_plus_smooth = np.full(n, np.nan)
        dm_minus_smooth = np.full(n, np.nan)
        
        if n >= period:
            # Initial average
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] > 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX
        adx = np.full(n, np.nan)
        for i in range(2*period-1, n):
            valid_dx = dx[period:i+1]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) >= period:
                adx[i] = np.nanmean(valid_dx[-period:])
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 30
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align indicators to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivots, ADX, and volume MA
    start_idx = max(5, vol_period, 28)  # 28 for ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        adx = adx_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly R3 + strong trend (ADX>25) + volume spike
            if (price > weekly_r3_aligned[i] and 
                adx > 25 and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly S3 + strong trend (ADX>25) + volume spike
            elif (price < weekly_s3_aligned[i] and 
                  adx > 25 and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price drops below weekly S3 OR trend weakens (ADX<20)
            if (price < weekly_s3_aligned[i] or 
                adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price rises above weekly R3 OR trend weakens (ADX<20)
            if (price > weekly_r3_aligned[i] or 
                adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_WeeklyPivot_R3S3_ADX25_Volume"
timeframe = "6h"
leverage = 1.0
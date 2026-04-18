#!/usr/bin/env python3
"""
12h Camarilla Pivot R3/S3 Breakout with Volume Confirmation and 1d ADX Trend Filter
- Uses 1d Camarilla pivot levels (R3/S3) as key support/resistance levels
- Long when price breaks above R3 with volume and ADX > 25 (trending)
- Short when price breaks below S3 with volume and ADX > 25 (trending)
- Exit when price returns to pivot point (PP) or ADX < 20 (range)
- Designed for 12h timeframe: 15-30 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper handling."""
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(len(close), np.nan)
    for i in range(period, len(tr)):
        if i == period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(close), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    for i in range(period, len(tr)):
        if i == period:
            atr[i] = np.nanmean(tr[1:i+1])
            dm_plus_smooth[i] = np.nanmean(dm_plus[1:i+1])
            dm_minus_smooth[i] = np.nanmean(dm_minus[1:i+1])
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(dm_plus_smooth), np.nan)
    minus_di = np.full(len(dm_minus_smooth), np.nan)
    dx = np.full(len(atr), np.nan)
    
    for i in range(len(tr)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx = np.full(len(dx), np.nan)
    for i in range(2*period-1, len(dx)):
        if i == 2*period-1:
            adx[i] = np.nanmean(dx[period:i+1])
        elif not np.isnan(dx[i]):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    # Typical price
    pp = (high + low + close) / 3
    range_val = high - low
    
    # Resistance levels
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    s3 = close - (range_val * 1.1 / 4)
    s4 = close - (range_val * 1.1 / 2)
    
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Camarilla levels on 1d
    pp_1d, r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align to 12h timeframe
    adx_1d_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    pp_1d_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate volume moving average (10-period)
    vol_ma = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_12h[i]) or np.isnan(pp_1d_12h[i]) or 
            np.isnan(r3_1d_12h[i]) or np.isnan(s3_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 10-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and ADX > 25 (trending)
            if close[i] > r3_1d_12h[i] and vol_confirmed and adx_1d_12h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and ADX > 25 (trending)
            elif close[i] < s3_1d_12h[i] and vol_confirmed and adx_1d_12h[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or ADX < 20 (range)
            if close[i] <= pp_1d_12h[i] or adx_1d_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or ADX < 20 (range)
            if close[i] >= pp_1d_12h[i] or adx_1d_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0
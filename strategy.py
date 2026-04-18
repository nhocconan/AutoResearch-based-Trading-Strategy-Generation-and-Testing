#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Volume_Strategy_v1
Hypothesis: Use daily Camarilla pivot levels (S1/S3 for long, R1/R3 for short) with volume confirmation and ADX trend filter. 
Go long when price crosses above S1 with volume > 1.5x average and ADX > 25, short when price crosses below R1 with same conditions. 
Exit on opposite pivot touch (S3/R3) or when ADX < 20 (range market). 
Designed for 12h timeframe to capture multi-day swings while minimizing trades (target 15-25/year). 
Works in bull markets via breakouts above R1, in bear via breakdowns below S1, and in ranging markets via mean reversion at S1/R1.
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
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 6)
    # S2 = C - (Range * 1.1 / 4)
    # S3 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 6)
    # R2 = C + (Range * 1.1 / 4)
    # R3 = C + (Range * 1.1 / 2)
    
    pivot_1d = np.full_like(high_1d, np.nan)
    range_1d = np.full_like(high_1d, np.nan)
    s1_1d = np.full_like(high_1d, np.nan)
    s3_1d = np.full_like(high_1d, np.nan)
    r1_1d = np.full_like(high_1d, np.nan)
    r3_1d = np.full_like(high_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pivot_1d[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        range_1d[i] = high_1d[i-1] - low_1d[i-1]
        s1_1d[i] = close_1d[i-1] - (range_1d[i] * 1.1 / 6)
        s3_1d[i] = close_1d[i-1] - (range_1d[i] * 1.1 / 2)
        r1_1d[i] = close_1d[i-1] + (range_1d[i] * 1.1 / 6)
        r3_1d[i] = close_1d[i-1] + (range_1d[i] * 1.1 / 2)
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        if len(tr) >= period + 1:
            # Initial averages
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_sum = np.nansum(plus_dm[1:period+1])
            minus_dm_sum = np.nansum(minus_dm[1:period+1])
            
            # Smoothing
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                plus_dm_sum = plus_dm_sum * (period - 1) / period + plus_dm[i]
                minus_dm_sum = minus_dm_sum * (period - 1) / period + minus_dm[i]
                
                plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        # ADX calculation
        dx = np.full_like(tr, np.nan)
        adx = np.full_like(tr, np.nan)
        
        for i in range(period + 1, len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if len(tr) >= 2 * period + 1:
            adx[2*period] = np.nanmean(dx[period+1:2*period+1])
            for i in range(2*period + 1, len(tr)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Align all 1d indicators to 12h timeframe
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30) + 1  # vol_period + adx stabilization
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume and ADX > 25 (trending)
            if close[i] > s1_1d_aligned[i] and close[i-1] <= s1_1d_aligned[i-1] and vol_confirm and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume and ADX > 25 (trending)
            elif close[i] < r1_1d_aligned[i] and close[i-1] >= r1_1d_aligned[i-1] and vol_confirm and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches S3 (strong support) or ADX < 20 (ranging)
            if close[i] <= s3_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches R3 (strong resistance) or ADX < 20 (ranging)
            if close[i] >= r3_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Volume_Strategy_v1"
timeframe = "12h"
leverage = 1.0
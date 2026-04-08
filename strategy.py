#!/usr/bin/env python3
# [24900] 4h_1d_camarilla_pivot_v1
# Hypothesis: 4-hour Camarilla pivot reversal with volume confirmation and 1-day ADX trend filter.
# Long when price touches or breaks below S3 with bullish divergence and ADX > 25 (trending market).
# Short when price touches or breaks above R3 with bearish divergence and ADX > 25.
# Exit when price reaches opposite pivot level (S1/R1) or volume drops below average.
# Uses Camarilla levels from daily timeframe for institutional reversal points.
# Designed to work in both bull and bear markets by filtering for trending conditions only.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_r2 = np.full_like(close_1d, np.nan)
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    camarilla_s2 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
            pivot = (ph + pl + pc) / 3.0
            range_val = ph - pl
            
            camarilla_r3[i] = pc + range_val * 1.1 / 2.0
            camarilla_r2[i] = pc + range_val * 1.1 / 4.0
            camarilla_r1[i] = pc + range_val * 1.1 / 6.0
            camarilla_s1[i] = pc - range_val * 1.1 / 6.0
            camarilla_s2[i] = pc - range_val * 1.1 / 4.0
            camarilla_s3[i] = pc - range_val * 1.1 / 2.0
    
    # Calculate 1-day ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        # Initial values
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full_like(tr, np.nan)
        di_minus = np.full_like(tr, np.nan)
        dx = np.full_like(tr, np.nan)
        
        for i in range(period, len(tr)):
            if atr[i] != 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX
        adx = np.full_like(tr, np.nan)
        if len(dx) >= 2*period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4-hour volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day indicators to 4-hour timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s1 = camarilla_s1_aligned[i]
        adx = adx_14_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price reaches S1 or volume drops below average
            if price <= s1 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches R1 or volume drops below average
            if price >= r1 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches/below S3 with volume expansion and ADX > 25
            if price <= s3 and vol_ratio > 1.3 and adx > 25:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches/above R3 with volume expansion and ADX > 25
            elif price >= r3 and vol_ratio > 1.3 and adx > 25:
                position = -1
                signals[i] = -0.25
    
    return signals
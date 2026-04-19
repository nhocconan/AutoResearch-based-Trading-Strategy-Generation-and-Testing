#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_Volume_ADX_Filter"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                else:
                    result[i] = np.nan
        return result
    
    atr = WilderSmooth(tr, period)
    dm_plus_smooth = WilderSmooth(dm_plus, period)
    dm_minus_smooth = WilderSmooth(dm_minus, period)
    
    dx = np.full_like(close, np.nan)
    mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
    dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
    
    adx = WilderSmooth(dx, period)
    return adx

def calculate_weekly_pivot(high, low, close):
    # Weekly pivot points using previous week's OHLC
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) for trend strength filter - calculated on 6h data
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    wh = df_1w['high'].shift(1).values  # Previous week high
    wl = df_1w['low'].shift(1).values   # Previous week low
    wc = df_1w['close'].shift(1).values # Previous week close
    
    # Calculate weekly pivot points
    wp, wr1, ws1, wr2, ws2, wr3, ws3 = calculate_weekly_pivot(wh, wl, wc)
    
    # Align weekly pivot points to 6h timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1w, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1w, ws2)
    wr3_aligned = align_htf_to_ltf(prices, df_1w, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1w, ws3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_6h[i]) or np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or 
            np.isnan(ws1_aligned[i]) or np.isnan(wr2_aligned[i]) or np.isnan(ws2_aligned[i]) or
            np.isnan(wr3_aligned[i]) or np.isnan(ws3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_6h[i] > 25
        
        if position == 0:
            # Long: price breaks above WR1 with volume and strong trend
            if (close[i] > wr1_aligned[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below WS1 with volume and strong trend
            elif (close[i] < ws1_aligned[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below WS1 or trend weakens (ADX < 20)
            if (close[i] < ws1_aligned[i]) or (adx_6h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above WR1 or trend weakens (ADX < 20)
            if (close[i] > wr1_aligned[i]) or (adx_6h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
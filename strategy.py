#!/usr/bin/env python3
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (based on previous week)
    P = np.full_like(high_1d, np.nan)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    R2 = np.full_like(high_1d, np.nan)
    S2 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        P[i] = pivot
        R1[i] = 2 * pivot - prev_low
        S1[i] = 2 * pivot - prev_high
        R2[i] = pivot + (prev_high - prev_low)
        S2[i] = pivot - (prev_high - prev_low)
    
    # Calculate 14-period ADX for trend filter
    def calculate_adx(high, low, close, period=14):
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
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            # Initial ADX
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Wilder smoothing for ADX
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Donchian channels (20-period) on 12h data
    upper = np.full_like(high_12h, np.nan)
    lower = np.full_like(low_12h, np.nan)
    
    for i in range(len(high_12h)):
        if i >= 19:  # 20-period lookback
            upper[i] = np.max(high_12h[i-19:i+1])
            lower[i] = np.min(low_12h[i-19:i+1])
    
    # Align all data to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_12h, upper)
    lower_6h = align_htf_to_ltf(prices, df_12h, lower)
    P_6h = align_htf_to_ltf(prices, df_1d, P)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average (48h lookback)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(P_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_6h[i] > 25
        
        if position == 0:
            # Long: breakout above upper Donchian with volume in trending market
            if close[i] > upper_6h[i] and vol_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian with volume in trending market
            elif close[i] < lower_6h[i] and vol_confirm and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below pivot OR ADX drops below 20 (losing momentum)
            if close[i] < P_6h[i] or adx_6h[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above pivot OR ADX drops below 20 (losing momentum)
            if close[i] > P_6h[i] or adx_6h[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0
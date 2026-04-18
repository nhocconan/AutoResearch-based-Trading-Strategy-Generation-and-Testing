#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Filtered
Hypothesis: Price breaks above/below Donchian channels with volume confirmation 
trigger trend-following entries. Uses 1d ADX filter to avoid choppy markets and 
1w EMA trend filter to align with higher timeframe momentum. Designed for low 
trade frequency (20-40/year) to minimize fee drag while capturing sustained 
trends in both bull and bear markets.
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
    
    # Donchian channel (20-period) on 4h
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n >= donchian_period:
        for i in range(donchian_period - 1, n):
            upper[i] = np.max(high[i - donchian_period + 1:i + 1])
            lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for regime filtering
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all higher timeframe data to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20, 34) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx_4h[i]) or np.isnan(ema_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filters: daily ADX > 25 (trending) AND price above weekly EMA (bullish bias)
        trending = adx_4h[i] > 25
        bullish_bias = close[i] > ema_1w_4h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume in trending market
            if close[i] > upper[i] and vol_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume in trending market
            elif close[i] < lower[i] and vol_confirm and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR ADX drops below 20 (losing trend)
            if close[i] < lower[i] or adx_4h[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR ADX drops below 20 (losing trend)
            if close[i] > upper[i] or adx_4h[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Filtered"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakouts and volume confirmation.
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean reversion), < 38.2 indicates trending.
# In trending markets (CHOP < 38.2), we trade Donchian breakouts with volume confirmation.
# In ranging markets (CHOP > 61.8), we fade the edges with volume confirmation.
# Uses 1d ADX for additional trend strength filter.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
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
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.zeros_like(high)
        dm_plus_smooth = np.zeros_like(high)
        dm_minus_smooth = np.zeros_like(high)
        
        # Initial values (simple average)
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(high)
        dx[period+1:] = 100 * np.abs(di_plus[period+1:] - di_minus[period+1:]) / (di_plus[period+1:] + di_minus[period+1:])
        
        adx = np.zeros_like(high)
        adx[2*period+1:] = np.nan
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Choppiness Index (14-period) on 4h
    def calculate_choppiness(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of TR over period
        tr_sum = np.zeros_like(high)
        for i in range(period, len(high)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(high)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(high)
        for i in range(period-1, len(high)):
            if tr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = np.nan
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        if len(high) < period:
            return np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        upper = np.zeros_like(high)
        lower = np.zeros_like(high)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        
        return upper, lower
    
    donch_up, donch_dn = calculate_donchian(high, low, 20)
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(chop[i]) or np.isnan(donch_up[i]) or np.isnan(donch_dn[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        adx_val = adx_1d_aligned[i]
        upper = donch_up[i]
        lower = donch_dn[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # Regime filters
        is_trending = chop_val < 38.2 and adx_val > 20
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long entry conditions
            long_signal = False
            if is_trending:
                # Breakout above upper Donchian in trending market
                if price > upper and volume_confirm:
                    long_signal = True
            elif is_ranging:
                # Mean reversion from lower Donchian in ranging market
                if price < lower and volume_confirm:
                    long_signal = True
            
            # Short entry conditions
            short_signal = False
            if is_trending:
                # Breakdown below lower Donchian in trending market
                if price < lower and volume_confirm:
                    short_signal = True
            elif is_ranging:
                # Mean reversion from upper Donchian in ranging market
                if price > upper and volume_confirm:
                    short_signal = True
            
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite Donchian touch or regime change to ranging with opposite signal
            if price < lower or (is_ranging and price > upper):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: opposite Donchian touch or regime change to ranging with opposite signal
            if price > upper or (is_ranging and price < lower):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0
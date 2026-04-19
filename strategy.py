#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h ADX trend filter.
# Works in bull by catching breakouts, in bear by avoiding false breakouts via ADX.
# Target: 20-40 trades/year per symbol, low turnover, high win rate.
name = "4h_Donchian20_Volume_ADX12h_Filter_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
        
        # Smooth TR, +DM, -DM
        atr = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        # Initial smoothed values
        atr[period-1] = np.sum(tr[1:period]) / period
        plus_dm_sum = np.sum(plus_dm[1:period]) / period
        minus_dm_sum = np.sum(minus_dm[1:period]) / period
        
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smoothed = (plus_di[i-1] * (period-1) + plus_dm[i]) / period if i < len(plus_di) else 0
            minus_dm_smoothed = (minus_di[i-1] * (period-1) + minus_dm[i]) / period if i < len(minus_di) else 0
            # Actually compute smoothed +DM and -DM properly
            plus_dm_smoothed = (plus_di[i-1] * (period-1) + plus_dm[i]) / period if i > 0 else plus_dm[i]
            minus_dm_smoothed = (minus_di[i-1] * (period-1) + minus_dm[i]) / period if i > 0 else minus_dm[i]
            # Fix: use separate arrays for smoothed DM
            if i == period:
                plus_dm_smoothed = np.sum(plus_dm[1:period]) / period
                minus_dm_smoothed = np.sum(minus_dm[1:period]) / period
            else:
                plus_dm_smoothed = (plus_dm_smoothed_prev * (period-1) + plus_dm[i]) / period
                minus_dm_smoothed = (minus_dm_smoothed_prev * (period-1) + minus_dm[i]) / period
            
            # Store for next iteration
            if i == period:
                plus_dm_smoothed_prev = plus_dm_smoothed
                minus_dm_smoothed_prev = minus_dm_smoothed
            elif i > period:
                plus_dm_smoothed_prev = plus_dm_smoothed
                minus_dm_smoothed_prev = minus_dm_smoothed
            
            plus_di[i] = 100 * plus_dm_smoothed / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smoothed / atr[i] if atr[i] != 0 else 0
        
        # Calculate DX and ADX
        dx = np.zeros(n)
        for i in range(period, n):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        adx = np.zeros(n)
        adx[2*period-1] = np.sum(dx[period:2*period]) / period
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate ADX with proper smoothing
    period = 14
    n_12h = len(high_12h)
    if n_12h < period * 2:
        adx_12h = np.full(n_12h, np.nan)
    else:
        tr = np.zeros(n_12h)
        plus_dm = np.zeros(n_12h)
        minus_dm = np.zeros(n_12h)
        
        for i in range(1, n_12h):
            hl = high_12h[i] - low_12h[i]
            hc = abs(high_12h[i] - close_12h[i-1])
            lc = abs(low_12h[i] - close_12h[i-1])
            tr[i] = max(hl, hc, lc)
            
            up = high_12h[i] - high_12h[i-1]
            down = low_12h[i-1] - low_12h[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
        
        # Smoothing
        atr = np.zeros(n_12h)
        plus_dm_smooth = np.zeros(n_12h)
        minus_dm_smooth = np.zeros(n_12h)
        
        # Initial values
        atr[period-1] = np.mean(tr[1:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
        
        for i in range(period, n_12h):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx_12h = np.zeros(n_12h)
        adx_12h[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n_12h):
            adx_12h[i] = (adx_12h[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Donchian channels on 4h (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # ADX trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long when price breaks above upper Donchian + volume spike + strong trend
            if close[i] > upper[i] and volume_spike[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian + volume spike + strong trend
            elif close[i] < lower[i] and volume_spike[i] and strong_trend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian or trend weakens
            if close[i] < lower[i] or adx_12h_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian or trend weakens
            if close[i] > upper[i] or adx_12h_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
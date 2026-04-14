#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ADX (14-period) for trend strength
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        n = len(close_arr)
        if n < period * 2:
            return np.full(n, np.nan)
        
        tr = np.zeros(n)
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
            dm_plus[i] = max(high_arr[i] - high_arr[i-1], 0)
            dm_minus[i] = max(low_arr[i-1] - low_arr[i], 0)
            dm_plus[i] = dm_plus[i] if dm_plus[i] > dm_minus[i] else 0
            dm_minus[i] = dm_minus[i] if dm_minus[i] > dm_plus[i] else 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing
        atr = np.zeros(n)
        dm_plus_smooth = np.zeros(n)
        dm_minus_smooth = np.zeros(n)
        
        atr[period-1] = np.sum(tr[1:period+1])
        dm_plus_smooth[period-1] = np.sum(dm_plus[1:period+1])
        dm_minus_smooth[period-1] = np.sum(dm_minus[1:period+1])
        
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        adx = np.full(n, np.nan)
        if n >= period * 2:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate daily RSI (14-period) for overbought/oversold
    def calculate_rsi(close_arr, period=14):
        n = len(close_arr)
        if n < period + 1:
            return np.full(n, np.nan)
        
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rsi = np.full(n, np.nan)
        for i in range(period+1, n):
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
        
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        if adx_1d_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # RSI filter: Avoid extreme overbought/oversold conditions
        if rsi_1d_aligned[i] > 75 or rsi_1d_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high in uptrend (RSI > 50)
            if close[i] > donch_high[i] and rsi_1d_aligned[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low in downtrend (RSI < 50)
            elif close[i] < donch_low[i] and rsi_1d_aligned[i] < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR RSI turns bearish
            if close[i] < donch_low[i] or rsi_1d_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR RSI turns bullish
            if close[i] > donch_high[i] or rsi_1d_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ADX_RSI_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0
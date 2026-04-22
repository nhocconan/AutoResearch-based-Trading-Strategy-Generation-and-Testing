#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly ADX trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and weekly ADX > 25 (trending) with volume spike.
Short when price breaks below lower Donchian channel and weekly ADX > 25 with volume spike.
Exit when price returns to the middle of the Donchian channel (mean reversion) or ADX < 20 (ranging).
Designed for low trade frequency by requiring breakout + trend + volume confluence.
Works in bull markets by catching breakouts and in bear markets by catching breakdowns.
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
    
    # Daily Donchian channel (20-period)
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        middle = np.full_like(high, np.nan, dtype=float)
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
            middle[i] = (upper[i] + lower[i]) / 2.0
        return upper, lower, middle
    
    upper, lower, middle = donchian_channels(high, low, 20)
    
    # Load weekly data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ADX (14-period)
    def calculate_adx(high, low, close, period):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        def wilders_smoothing(arr, period):
            smoothed = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return smoothed
            smoothed[period - 1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + arr[i]
            return smoothed
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.full_like(dx, np.nan, dtype=float)
        # First ADX is average of first 'period' DX values
        if len(dx) >= 2 * period:
            adx[2 * period - 1] = np.nanmean(dx[period:2 * period])
            for i in range(2 * period, len(dx)):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        return adx
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above upper Donchian + ADX > 25 (strong trend) + volume spike
            if close[i] > upper[i] and adx_1w_aligned[i] > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + ADX > 25 + volume spike
            elif close[i] < lower[i] and adx_1w_aligned[i] > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle of channel OR ADX < 20 (ranging market)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle or trend weakens
                if close[i] <= middle[i] or adx_1w_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle or trend weakens
                if close[i] >= middle[i] or adx_1w_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "DailyDonchian20_WeeklyADX25_Volume"
timeframe = "1d"
leverage = 1.0
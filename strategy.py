#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Donchian channels with 1-day volume confirmation.
# Long when price breaks above weekly Donchian high with daily volume > 2x 20-day average.
# Short when price breaks below weekly Donchian low with daily volume > 2x 20-day average.
# Uses 1-week ADX(14) > 25 to ensure only strong trends are traded, avoiding chop.
# Designed for low trade frequency (15-30/year) to capture strong trend continuations.
# Works in bull markets (breaks highs) and bear markets (breaks lows) due to symmetric logic.

name = "6h_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donch_high = np.full_like(close_1w, np.nan)
    donch_low = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        donch_high[i] = np.max(high_1w[i-20:i])
        donch_low[i] = np.min(low_1w[i-20:i])
    
    # Weekly ADX(14) for trend strength
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
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.full_like(high, np.nan)
        dm_plus_smooth = np.full_like(high, np.nan)
        dm_minus_smooth = np.full_like(high, np.nan)
        
        # Initial averages
        if len(high) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder smoothing
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DX and ADX
        dx = np.full_like(high, np.nan)
        adx = np.full_like(high, np.nan)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                dx[i] = (np.abs(dm_plus_smooth[i] - dm_minus_smooth[i]) / atr[i]) * 100
        
        if len(high) >= 2 * period - 1:
            adx[2*period-2] = np.nanmean(dx[period:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    strong_trend = (adx_1w >= 25).astype(float)  # 1.0 when ADX >= 25
    
    # Align weekly indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Daily volume EMA(20)
    vol_ema_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ema_1d  # Current volume vs 20-day average
    
    # Align daily volume ratio to 6h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(strong_trend_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above weekly Donchian high in strong trend with volume spike
            if (strong_trend_aligned[i] > 0.5 and  # Strong trend (ADX >= 25)
                close[i] > donch_high_aligned[i] and  # Break above Donchian high
                vol_ratio_aligned[i] > 2.0):          # Daily volume > 2x 20-day average
                signals[i] = 0.25
                position = 1
            # Short setup: break below weekly Donchian low in strong trend with volume spike
            elif (strong_trend_aligned[i] > 0.5 and  # Strong trend (ADX >= 25)
                  close[i] < donch_low_aligned[i] and   # Break below Donchian low
                  vol_ratio_aligned[i] > 2.0):          # Daily volume > 2x 20-day average
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below weekly Donchian low or trend weakens
            if close[i] < donch_low_aligned[i] or strong_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above weekly Donchian high or trend weakens
            if close[i] > donch_high_aligned[i] or strong_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
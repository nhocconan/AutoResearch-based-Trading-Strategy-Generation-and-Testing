#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla pivot breakout (R3/S3) filtered by 1d ADX trend strength (>25) and 1d volume spike (>2x 20-day average).
Long on breakout above R3 in bullish 1d trend with volume spike.
Short on breakout below S3 in bearish 1d trend with volume spike.
Exit on opposite Camarilla level (R2/S2) touch.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3, R2, S2) using previous 12h candle
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous period's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_R3[i] = pc + (range_val * 1.1000 / 4)
        camarilla_S3[i] = pc - (range_val * 1.1000 / 4)
        camarilla_R2[i] = pc + (range_val * 1.1000 / 6)
        camarilla_S2[i] = pc - (range_val * 1.1000 / 6)
    
    # 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder smoothing)
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_period = 14
    atr = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, atr_period)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume spike condition: current 1d volume > 2x 20-day average
    vol_spike = vol_1d > (2 * vol_ma_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(camarilla_R2[i]) or np.isnan(camarilla_S2[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend direction using ADX and price vs 20-period SMA
        sma_20_1d = np.full_like(close_1d, np.nan)
        for j in range(20, len(close_1d)):
            sma_20_1d[j] = np.mean(close_1d[j-20:j])
        sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
        
        if not np.isnan(sma_20_1d_aligned[i]):
            trend_1d_up = adx_aligned[i] > 25 and close_1d_aligned[i] > sma_20_1d_aligned[i]
            trend_1d_down = adx_aligned[i] > 25 and close_1d_aligned[i] < sma_20_1d_aligned[i]
        else:
            trend_1d_up = False
            trend_1d_down = False
        
        if position == 0:
            # Long: Camarilla R3 breakout in 1d uptrend with volume spike
            if (close[i] > camarilla_R3[i] and 
                trend_1d_up and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakdown in 1d downtrend with volume spike
            elif (close[i] < camarilla_S3[i] and 
                  trend_1d_down and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below camarilla S2
            if close[i] < camarilla_S2[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above camarilla R2
            if close[i] > camarilla_R2[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
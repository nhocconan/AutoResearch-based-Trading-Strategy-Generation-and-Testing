#!/usr/bin/env python3

# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: 12h Camarilla R3/S3 breakout filtered by 1w ADX trend (>25) and volume spike (>2x 10-week average).
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 15-35 trades/year per symbol to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get 1w data for filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
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
    
    # 1w ADX for trend filter (10-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
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
    
    atr_period = 10
    atr = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, atr_period)
    
    # 1w volume average (10-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full_like(vol_1w, np.nan)
    for i in range(10, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-10:i])
    
    # Align 1w indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume spike condition: current 1w volume > 2x 10-week average
    vol_spike = vol_1w > (2 * vol_ma_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # Prevent overtrading (approx 6 days)
    
    start_idx = max(10, 20)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(camarilla_R2[i]) or np.isnan(camarilla_S2[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1w trend direction using ADX and price vs 10-period SMA
        sma_10_1w = np.full_like(close_1w, np.nan)
        for j in range(10, len(close_1w)):
            sma_10_1w[j] = np.mean(close_1w[j-10:j])
        sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
        
        if not np.isnan(sma_10_1w_aligned[i]):
            trend_1w_up = adx_aligned[i] > 25 and close_1w_aligned[i] > sma_10_1w_aligned[i]
            trend_1w_down = adx_aligned[i] > 25 and close_1w_aligned[i] < sma_10_1w_aligned[i]
        else:
            trend_1w_up = False
            trend_1w_down = False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Camarilla R3 breakout in 1w uptrend with volume spike
            if (close[i] > camarilla_R3[i] and 
                trend_1w_up and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Camarilla S3 breakdown in 1w downtrend with volume spike
            elif (close[i] < camarilla_S3[i] and 
                  trend_1w_down and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below camarilla S2
            if close[i] < camarilla_S2[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above camarilla R2
            if close[i] > camarilla_R2[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals
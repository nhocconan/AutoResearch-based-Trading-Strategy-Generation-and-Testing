#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 mean reversion with 1d ADX regime filter
# Long when price touches/below S3 and 1d ADX < 25 (range market)
# Short when price touches/above R3 and 1d ADX < 25 (range market)
# Uses volume confirmation (>1.5x 20-period avg) to filter false touches
# Works in ranging markets (BTC/ETH 2022-2024, 2025+) by fading extremes
# Discrete position sizing (0.25) controls drawdown and minimizes fee churn
# Target: 50-150 total trades over 4 years on 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial values (first 14 periods)
    tr_sum[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_sum[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_sum[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    
    # ADX smoothing (first valid ADX at index 2*tr_period)
    adx_start = 2 * tr_period
    if len(dx) > adx_start:
        adx[adx_start] = np.nanmean(dx[tr_period+1:adx_start+1])
        for i in range(adx_start + 1, len(dx)):
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Camarilla Pivot Levels (R3, S3) ===
    # Based on previous day's OHLC
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    # We use daily OHLC to compute Camarilla for 6h bars
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels
    camarilla_r3 = daily_close + ((daily_high - daily_low) * 1.1 / 4)
    camarilla_s3 = daily_close - ((daily_high - daily_low) * 1.1 / 4)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # ADX(14) needs ~28, Donchian(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: 1d ADX < 25 (range-bound market)
        range_market = adx_1d_aligned[i] < 25
        
        # === LONG CONDITIONS ===
        # 1. Price touches/below S3 (low <= S3)
        # 2. Range market (ADX < 25)
        # 3. Volume confirmation
        if (low[i] <= s3_aligned[i]) and range_market and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price touches/above R3 (high >= R3)
        # 2. Range market (ADX < 25)
        # 3. Volume confirmation
        elif (high[i] >= r3_aligned[i]) and range_market and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_1dADX_Range_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0
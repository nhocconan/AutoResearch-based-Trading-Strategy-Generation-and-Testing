#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w ADX25 trend filter + 1d volume spike
# Donchian breakouts capture strong momentum moves. Weekly ADX > 25 ensures we only
# trade in strong trending regimes (works in both bull and bear markets via direction).
# Daily volume confirmation filters false breakouts. Designed for 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn. Works on BTC/ETH via regime alignment.

name = "6h_Donchian20_1wADX25_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for ADX25 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) with smoothing period 25
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.full_like(tr, np.nan)
    dm_plus_sum = np.full_like(dm_plus, np.nan)
    dm_minus_sum = np.full_like(dm_minus, np.nan)
    
    # Wilder's smoothing
    for i in range(len(tr)):
        if i < tr_period:
            continue
        if i == tr_period:
            tr_sum[i] = np.nansum(tr[i-tr_period+1:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-tr_period+1:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-tr_period+1:i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    # ADX: smoothed DX over 25 periods
    adx = np.full_like(dx, np.nan)
    adx_period = 25
    for i in range(len(dx)):
        if i < adx_period:
            continue
        if i == adx_period:
            adx[i] = np.nanmean(dx[i-adx_period+1:i+1])
        else:
            if np.isnan(adx[i-1]):
                adx[i] = np.nanmean(dx[i-adx_period+1:i+1])
            else:
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    adx_25 = adx  # ADX with 25-period smoothing
    adx_25_aligned = align_htf_to_ltf(prices, df_1w, adx_25)
    
    # Get 1d data for volume EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume EMA(20)
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            continue
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start from lookback to have valid Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_25_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly ADX > 25 indicates strong trend
        strong_trend = adx_25_aligned[i] > 25
        
        # Daily volume spike: current volume > 2.0 * 1d volume EMA(20)
        volume_spike = volume[i] > (2.0 * vol_ema_20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper band in strong uptrend with volume spike
            if close[i] > highest_high[i] and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band in strong downtrend with volume spike
            elif close[i] < lowest_low[i] and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower band or loses strong trend
            if close[i] < lowest_low[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper band or loses strong trend
            if close[i] > highest_high[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
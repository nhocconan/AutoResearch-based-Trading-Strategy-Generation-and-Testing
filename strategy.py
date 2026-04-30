#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX25 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -90 or > -10) with
# volume spike indicate potential reversal. 1d ADX > 25 ensures we only trade in strong trends
# to avoid whipsaw in ranging markets. Volume spike (2.0x 24-period average) confirms
# institutional participation. Discrete sizing 0.25 minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull markets via
# oversold bounces in uptrends and bear markets via overbought reversals in downtrends.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original array
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]),
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_period = len(tr)
    atr = np.full(tr_period, np.nan)
    dm_plus_smooth = np.full(tr_period, np.nan)
    dm_minus_smooth = np.full(tr_period, np.nan)
    
    # Initial values (simple average)
    if tr_period >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        
        # Wilder's smoothing
        for i in range(period, tr_period):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full(tr_period, np.nan)
    di_minus = np.full(tr_period, np.nan)
    dx = np.full(tr_period, np.nan)
    
    for i in range(period-1, tr_period):
        if not np.isnan(atr[i]) and atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # ADX: smoothed DX
    adx = np.full(tr_period, np.nan)
    if tr_period >= 2*period-1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, tr_period):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d ADX to 6h timeframe (needs extra delay for ADX confirmation)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    
    # Williams %R on 6h data (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*6h = 144h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 24, 2*14)  # warmup for Williams %R, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_adx = adx_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require ADX > 25 (strong trend) and volume spike
            if curr_adx > 25 and curr_volume_spike:
                # Bullish entry: Williams %R < -90 (oversold) in uptrend
                if curr_williams_r < -90:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -10 (overbought) in downtrend
                elif curr_williams_r > -10:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading) OR ADX drops below 20
            if curr_williams_r > -50 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading) OR ADX drops below 20
            if curr_williams_r < -50 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
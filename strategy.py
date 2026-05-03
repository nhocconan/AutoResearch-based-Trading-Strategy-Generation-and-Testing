#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Camarilla levels provide precise intraday support/resistance, volume spike confirms
# institutional participation, ADX>25 ensures we trade only in trending markets to
# avoid whipsaws in ranging conditions. Designed for low trade frequency (<40/year)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/14)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We focus on R3/S3 as stronger breakout levels
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    # For each 4h bar, calculate Camarilla levels from the most recent completed 1d bar
    for i in range(n):
        # Find the index of the most recent completed 1d bar
        # Since we're on 4h timeframe, we can use the previous 1d bar's data
        # We'll compute this efficiently by tracking the last 1d bar close time
        if i < 6:  # Need at least 6*4h = 24h to get first 1d bar
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            continue
            
        # Get the most recent completed 1d bar's OHLC
        # We approximate by using the 1d bar that closed at or before this 4h bar
        # In practice, we use the previous day's OHLC for today's Camarilla levels
        # For simplicity, we'll use a rolling window approach on 1d data aligned to 4h
        
        # Instead, we calculate Camarilla levels once per 1d bar and align
        # This is more efficient and correct
        pass  # We'll calculate below using vectorized approach
    
    # Vectorized approach: calculate Camarilla levels for each 1d bar, then align
    # Camarilla levels based on previous day's OHLC
    if len(df_1d) >= 2:
        # Previous day's OHLC
        prev_high = df_1d['high'].values[:-1]  # Shifted by 1
        prev_low = df_1d['low'].values[:-1]
        prev_close = df_1d['close'].values[:-1]
        
        # Camarilla levels
        camarilla_pp = (prev_high + prev_low + prev_close) / 3
        camarilla_r3 = camarilla_pp + (prev_high - prev_low) * 1.1 / 4
        camarilla_s3 = camarilla_pp - (prev_high - prev_low) * 1.1 / 4
        
        # Prepend NaN for first day (no previous day)
        camarilla_r3_1d = np.concatenate([[np.nan], camarilla_r3])
        camarilla_s3_1d = np.concatenate([[np.nan], camarilla_s3])
        
        # Align to 4h timeframe
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Break above R3 level + volume spike + trending market
            if high[i] > camarilla_r3_aligned[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 level + volume spike + trending market
            elif low[i] < camarilla_s3_aligned[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price re-enters below R3 level OR reverse signal
            if low[i] < camarilla_r3_aligned[i] or (low[i] < camarilla_s3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price re-enters above S3 level OR reverse signal
            if high[i] > camarilla_s3_aligned[i] or (high[i] > camarilla_r3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
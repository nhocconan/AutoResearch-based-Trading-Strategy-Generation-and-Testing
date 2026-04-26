#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 12h timeframe, use Camarilla R3/S3 levels from 1d pivot points for breakout entries with 1d trend filter (close > 1d EMA34) and volume confirmation (>2.0x 20-period average). Add ADX regime filter (ADX > 25) to avoid whipsaws in ranging markets. Exit on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal. Designed for 12-37 trades/year on 12h by requiring multi-timeframe alignment and strict confluence, reducing fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    # Using previous day's values (shifted by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN (no previous day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ADX regime filter (ADX > 25 = trending market)
    # Calculate ADX using 14-period
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    atr[period_adx] = np.mean(tr[1:period_adx+1])  # Initial ATR
    
    for i in range(period_adx+1, n):
        atr[i] = (atr[i-1] * (period_adx-1) + tr[i]) / period_adx
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    # Initial smoothed values
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    
    plus_dm_smooth[period_adx] = np.sum(plus_dm[1:period_adx+1])
    minus_dm_smooth[period_adx] = np.sum(minus_dm[1:period_adx+1])
    
    for i in range(period_adx+1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period_adx) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period_adx) + minus_dm[i]
    
    plus_di[period_adx:] = 100 * plus_dm_smooth[period_adx:] / np.maximum(atr[period_adx:], 1e-10)
    minus_di[period_adx:] = 100 * minus_dm_smooth[period_adx:] / np.maximum(atr[period_adx:], 1e-10)
    
    dx[period_adx:] = 100 * np.abs(plus_di[period_adx:] - minus_di[period_adx:]) / np.maximum(plus_di[period_adx:] + minus_di[period_adx:], 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(high)
    adx[2*period_adx] = np.mean(dx[period_adx:2*period_adx+1])  # Initial ADX
    
    for i in range(2*period_adx+1, n):
        adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d EMA34) + volume MA + ADX warmup
    start_idx = max(34, 20, 2*14+14)  # 42 for ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(adx_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + volume spike + ADX filter
            long_signal = (close[i] > camarilla_r3_aligned[i]) and trend_1d_uptrend and volume_spike[i] and adx_filter[i]
            
            # Short: price breaks below S3 + 1d downtrend + volume spike + ADX filter
            short_signal = (close[i] < camarilla_s3_aligned[i]) and trend_1d_downtrend and volume_spike[i] and adx_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S3 OR 1d trend turns down OR ADX weakens
            if (close[i] < camarilla_s3_aligned[i] or not trend_1d_uptrend or not adx_filter[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R3 OR 1d trend turns up OR ADX weakens
            if (close[i] > camarilla_r3_aligned[i] or not trend_1d_downtrend or not adx_filter[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0
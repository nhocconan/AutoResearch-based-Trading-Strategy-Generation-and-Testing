#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/continuation points. 
# 1d EMA34 ensures alignment with daily trend. Volume spike confirms institutional participation.
# Designed for 75-200 total trades over 4 years (19-50/year) with strong performance in both bull and bear markets.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need previous day's high, low, close for today's levels
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)
    
    # Shift 1d data by 1 bar to get previous completed day
    df_1d_shift = df_1d.copy()
    df_1d_shift['high'] = df_1d['high'].shift(1)
    df_1d_shift['low'] = df_1d['low'].shift(1)
    df_1d_shift['close'] = df_1d['close'].shift(1)
    
    # Align the previous day's OHLC to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_shift, df_1d_shift['high'].values)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_shift, df_1d_shift['low'].values)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_shift, df_1d_shift['close'].values)
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or np.isnan(prev_close_aligned[i])):
            rng = prev_high_aligned[i] - prev_low_aligned[i]
            camarilla_r3[i] = prev_close_aligned[i] + rng * 1.1 / 4
            camarilla_s3[i] = prev_close_aligned[i] - rng * 1.1 / 4
            camarilla_r4[i] = prev_close_aligned[i] + rng * 1.1 / 2
            camarilla_s4[i] = prev_close_aligned[i] - rng * 1.1 / 2
    
    # Volume confirmation: 20-period EMA on 4h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 with uptrend alignment and volume spike
            if close[i] > camarilla_r3[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with downtrend alignment and volume spike
            elif close[i] < camarilla_s3[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend alignment
            if close[i] < camarilla_s3[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend alignment
            if close[i] > camarilla_r3[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance; breakouts capture momentum.
# 4h EMA50 ensures alignment with higher timeframe trend, reducing false signals in ranging markets.
# Volume confirmation (>1.5x 20-period EMA) filters low-momentum breakouts.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Designed for 60-150 total trades over 4 years (15-37/year) with discrete sizing (0.20) to minimize fee drag.
# Works in both bull and bear markets by following 4h trend direction and using Camarilla structure.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3, R4, S4 (using prior day's range)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (using prior completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMA and volume
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 4h EMA50 + volume spike
            if close[i] > camarilla_r3_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + below 4h EMA50 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 4h trend alignment
            if close[i] < camarilla_s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 4h trend alignment
            if close[i] > camarilla_r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
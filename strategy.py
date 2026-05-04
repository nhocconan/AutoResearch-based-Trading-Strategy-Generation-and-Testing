#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume spike confirmation and ATR-based trend filter
# Uses 1d Camarilla levels for institutional support/resistance
# Uses 4h ATR ratio (current/20-period) > 1.2 to ensure sufficient momentum (avoid chop)
# Uses 4h volume > 1.5x 20-period EMA for confirmation
# Designed for 4h timeframe targeting 25-35 trades/year with discrete sizing (0.25)
# Volume spike + ATR filter reduces false breakouts while capturing strong momentum moves
# Works in bull markets (breakouts with volume) and bear markets (trend continuation signals)

name = "4h_Camarilla_R3S3_Breakout_VolumeSpike_ATR12"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for ATR and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ATR for trend filter (period=14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Wilder's ATR smoothing
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            first_avg = np.nansum(values[1:period+1])
            result[period] = first_avg
            for i in range(period+1, len(values)):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    
    # Calculate 4h ATR EMA(20) for normalization
    atr_series = pd.Series(atr)
    atr_ema_20 = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema_20_aligned = align_htf_to_ltf(prices, df_4h, atr_ema_20)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_series = pd.Series(vol_4h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    # Calculate ATR ratio (current ATR / ATR EMA) for momentum filter
    atr_ratio = np.where(atr_ema_20_aligned > 0, atr / atr_ema_20_aligned, 1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_4h, atr_ratio)
    
    # Calculate camarilla levels: R3, S3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # Momentum confirmation: ATR ratio > 1.2 (sufficient momentum)
        momentum_confirmed = atr_ratio_aligned[i] > 1.2
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + momentum confirmation
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                momentum_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3 + volume confirmation + momentum confirmation
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  momentum_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
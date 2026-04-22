#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. ADX filters for trending regimes.
# In trending markets (ADX > 25), we take Williams %R reversals in the direction of trend.
# Works in both bull/bear markets by following higher timeframe trend.
# Uses discrete position sizing (0.25) to minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # ADX (14-period)
    # +DM = max(High - Previous High, 0) if High - Previous High > Previous Low - Low else 0
    # -DM = max(Previous Low - Low, 0) if Previous Low - Low > High - Previous High else 0
    # TR = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
    # +DM smoothed, -DM smoothed, TR smoothed
    # DI+ = 100 * smoothed +DM / smoothed TR
    # DI- = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    # ADX = smoothed DX
    
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    period = 14
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    di_plus = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    di_minus = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + ADX trending (> 25) + volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + ADX trending (> 25) + volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses midpoint (-50) in opposite direction
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if williams_r_aligned[i] > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if williams_r_aligned[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADX_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0
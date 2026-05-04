#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX Trend Filter and Volume Confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -90 or > -10) 
# often precede reversals. In strong trends (ADX > 25 on 1d), these extremes can offer 
# high-probability pullback entries. Volume confirmation ensures institutional participation.
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-25 trades/year.

name = "6h_WilliamsR_Extreme_1dADX_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d for trend strength
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no prior close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 != 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Calculate Williams %R (14-period) on 1d
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = np.full_like(close_1d, np.nan)
    lowest_low_14 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 13:  # Need 14 periods (0 to 13 inclusive)
            start_idx = i - 13
            highest_high_14[i] = np.max(high_1d[start_idx:i+1])
            lowest_low_14[i] = np.min(low_1d[start_idx:i+1])
    
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100, 
                          -50)  # Neutral when no range
    
    # Extreme levels: %R < -90 (oversold) or %R > -10 (overbought)
    oversold = williams_r < -90
    overbought = williams_r > -10
    
    # Align 1d indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    oversold_aligned = align_htf_to_ltf(prices, df_1d, oversold.astype(float))
    overbought_aligned = align_htf_to_ltf(prices, df_1d, overbought.astype(float))
    
    # Volume confirmation: 6h volume > 1.5x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(strong_trend_aligned[i]) or np.isnan(oversold_aligned[i]) or 
            np.isnan(overbought_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -90) AND strong trend AND volume spike
            if (oversold_aligned[i] > 0.5 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -10) AND strong trend AND volume spike
            elif (overbought_aligned[i] > 0.5 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum fading) OR trend weakens
            if (overbought_aligned[i] > 0.5 or  # Actually exiting when overbought signal appears
                strong_trend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 OR trend weakens
            if (oversold_aligned[i] > 0.5 or  # Actually exiting when oversold signal appears
                strong_trend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
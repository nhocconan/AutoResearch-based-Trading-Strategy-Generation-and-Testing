#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and 1d ADX < 25 (range/weak trend) with volume > 1.3x 20-bar average.
# Short when Williams %R > -20 (overbought) and 1d ADX < 25 with volume > 1.3x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies overextended moves; ADX filter avoids strong trends where mean reversion fails.
# Works in bull markets via mean reversion at extremes and in bear markets via fading spikes during low ADX regimes.

name = "6h_WilliamsR_MeanReversion_1dADX25_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    lookback = 14  # for Williams %R and volume average
    
    # Calculate Williams %R on primary timeframe
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < lookback:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[1:])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]), 
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]), 
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nanmean(values[:period])
        # Subsequent values: smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr = wilders_smoothing(tr, lookback)
    dm_plus_smooth = wilders_smoothing(dm_plus, lookback)
    dm_minus_smooth = wilders_smoothing(dm_minus, lookback)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, lookback)
    
    # Align 1d ADX to 6h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold), ADX < 25 (weak trend), volume spike
            if (williams_r[i] < -80 and 
                adx_aligned[i] < 25 and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought), ADX < 25 (weak trend), volume spike
            elif (williams_r[i] > -20 and 
                  adx_aligned[i] < 25 and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (recovery) OR ADX > 30 (strong trend emerging)
            if (williams_r[i] > -50 or 
                adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (decline) OR ADX > 30 (strong trend emerging)
            if (williams_r[i] < -50 or 
                adx_aligned[i] > 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
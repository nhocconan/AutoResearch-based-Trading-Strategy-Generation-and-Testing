#!/usr/bin/env python3
# Hypothesis: 4h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold) in low volatility regime (ADX < 25) with volume > 1.2x average.
# Short when Williams %R crosses below -20 (overbought) in low volatility regime (ADX < 25) with volume > 1.2x average.
# Uses discrete sizing 0.25 to target 50-100 total trades over 4 years on 4h timeframe.
# Williams %R identifies exhaustion points; ADX filter avoids whipsaws in strong trends; volume confirmation ensures participation.
# Works in bull markets via mean reversion at pullbacks and in bear markets via bounces in ranging conditions.

name = "4h_WilliamsR_MeanReversion_1dADX25_VolumeConfirm"
timeframe = "4h"
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
    
    # Williams %R period
    lookback = 14
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d data
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    # Initial TR, DM+, DM- sums
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Wilder's smoothing
    for i in range(len(tr)):
        if i == 0:
            tr_sum[i] = tr[i]
            dm_plus_sum[i] = dm_plus[i]
            dm_minus_sum[i] = dm_minus[i]
        else:
            tr_sum[i] = tr_sum[i-1] * (1 - alpha) + tr[i] * alpha
            dm_plus_sum[i] = dm_plus_sum[i-1] * (1 - alpha) + dm_plus[i] * alpha
            dm_minus_sum[i] = dm_minus_sum[i-1] * (1 - alpha) + dm_minus[i] * alpha
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / (tr_sum + 1e-10)
    di_minus = 100 * dm_minus_sum / (tr_sum + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period_adx:
            adx[i] = np.nan
        elif i == period_adx:
            adx[i] = np.nanmean(dx[1:i+1])  # First ADX is average of first period_adx DX values
        else:
            adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha
    
    # Align 1d ADX to 4h timeframe (wait for 1d bar to close)
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
            # LONG: Williams %R crosses above -80 (from below) in low volatility with volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                adx_aligned[i] < 25 and 
                volume[i] > 1.2 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (from above) in low volatility with volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  adx_aligned[i] < 25 and 
                  volume[i] > 1.2 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) OR volatility increases (ADX > 30)
            if (williams_r[i] > -20 and williams_r[i-1] <= -20) or adx_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) OR volatility increases (ADX > 30)
            if (williams_r[i] < -80 and williams_r[i-1] >= -80) or adx_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
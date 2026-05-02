#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX regime filter + volume spike
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag
# Williams %R identifies overbought/oversold conditions (long when < -80, short when > -20)
# 1d ADX > 25 ensures we only trade in trending markets (avoid chop)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Uses discrete position sizing 0.25 to balance exposure and risk
# Works in both bull and bear: ADX regime filter prevents whipsaw in ranging markets

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R (14-period)
    def williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * ((highest_high - close) / (highest_high - lowest_low)), 
                      -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (wr[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (wr[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR ADX < 20 (losing trend)
            if (wr[i] > -20 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR ADX < 20 (losing trend)
            if (wr[i] < -80 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
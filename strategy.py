#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d Volume Spike + ADX Regime Filter
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume spike (2.0x 20-bar MA)
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume spike
# Williams %R identifies exhaustion points; ADX ensures we only trade in trending markets to avoid chop
# Volume spike confirms institutional participation at turning points
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 80-160 total trades over 4 years (20-40/year) to balance signal quality and fee drag

name = "4h_WilliamsR_1dADX_Regime_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Williams %R (14-period)
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return to neutral) OR ADX <= 25 (trend weakening)
            if williams_r[i] > -50 or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return to neutral) OR ADX <= 25 (trend weakening)
            if williams_r[i] < -50 or adx_1d_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
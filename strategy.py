#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Trend + Volume Confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) with
# volume spike indicate potential reversals. 1d ADX > 25 ensures trades align with strong trend
# to avoid false signals in ranging markets. Designed for 50-150 total trades over 4 years (12-37/year)
# on 6h timeframe. Works in bull markets (buying oversold in uptrend) and bear markets
# (selling overbought in downtrend) by only taking trades in direction of 1d ADX trend.

name = "6h_WilliamsR_Extreme_1dADX_Volume"
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
    
    # Calculate 1d ADX for trend filter (min_periods=14 for ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 != 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data (lookback=14)
    def williams_r(high_arr, low_arr, close_arr, lookback=14):
        wr = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(lookback-1, len(close_arr)):
            highest_high = np.max(high_arr[i-lookback+1:i+1])
            lowest_low = np.min(low_arr[i-lookback+1:i+1])
            if highest_high != lowest_low:
                wr[i] = (highest_high - close_arr[i]) / (highest_high - lowest_low) * -100
            else:
                wr[i] = -50
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_14_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) with volume spike AND ADX > 25 (strong trend)
            if (wr[i] < -80 and 
                volume_spike[i] and 
                adx_14_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) with volume spike AND ADX > 25 (strong trend)
            elif (wr[i] > -20 and 
                  volume_spike[i] and 
                  adx_14_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum weakening) OR ADX < 20 (trend weakening)
            if wr[i] > -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum weakening) OR ADX < 20 (trend weakening)
            if wr[i] < -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
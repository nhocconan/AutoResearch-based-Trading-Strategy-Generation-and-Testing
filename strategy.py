#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND ADX(1d) > 25 (trending) AND volume > 1.5x 20-period average
# Short when Bear Power > 0 AND ADX(1d) > 25 (trending) AND volume > 1.5x 20-period average
# Exit when Bull/Bear Power <= 0 OR ADX(1d) < 20 (range) OR volume normalizes
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Elder Ray measures bull/bear strength relative to EMA, ADX filters for trending markets only,
# volume confirmation ensures institutional participation. Works in bull markets via strong Bull Power,
# in bear markets via strong Bear Power. Avoids whipsaws in ranging markets via ADX regime filter.

name = "6h_ElderRay_Power_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    if len(close) < 13:
        return np.zeros(n)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (Average Directional Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period]) / period
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr, atr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, atr_period)
    
    # Avoid division by zero
    dm_plus_smooth_safe = np.where(dm_plus_smoothed == 0, 1e-10, dm_plus_smoothed)
    dm_minus_smooth_safe = np.where(dm_minus_smoothed == 0, 1e-10, dm_minus_smoothed)
    tr_smooth_safe = np.where(tr_smoothed == 0, 1e-10, tr_smoothed)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # ADX is smoothed DX
    adx_period = 14
    adx = wilders_smoothing(dx, adx_period)
    # For first adx_period values, ADX is undefined, set to 0
    adx[:adx_period-1] = 0
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX > 25 (trending) AND volume spike
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND ADX > 25 (trending) AND volume spike
            elif (bear_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR ADX < 20 (range) OR volume normalizes
            if (bull_power[i] <= 0 or 
                adx_aligned[i] < 20 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR ADX < 20 (range) OR volume normalizes
            if (bear_power[i] <= 0 or 
                adx_aligned[i] < 20 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
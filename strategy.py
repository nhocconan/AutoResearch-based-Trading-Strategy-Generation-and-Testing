#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d Volume Spike and ADX Regime Filter
# Long when: Williams %R(14) < -80 (oversold) AND 1d volume > 1.8x 20-period average AND ADX(14) < 20 (low trend = range)
# Short when: Williams %R(14) > -20 (overbought) AND 1d volume > 1.8x 20-period average AND ADX(14) < 20 (low trend = range)
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) or opposite extreme
# Williams %R identifies exhaustion points in ranging markets
# Volume spike confirms institutional participation at extremes
# ADX < 20 ensures we trade in non-trending (range) markets where reversals work
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "6h_WilliamsR_Extreme_Reversal_1dVolumeSpike_ADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d ADX (14-period) for regime filter
    # ADX requires +DI and -DI calculation
    # First calculate True Range and Directional Movement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Handle first values
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period and not np.isnan(data[period-1]):
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    minus_di = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * vol_ma_20)
    
    # Align all 1d indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        adx_cond = bool(adx_aligned[i] < 20)  # Low trend = range market
        
        if position == 0:
            # Long: Oversold (%R < -80) in range with volume spike
            if williams_r_aligned[i] < -80 and vol_cond and adx_cond:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (%R > -20) in range with volume spike
            elif williams_r_aligned[i] > -20 and vol_cond and adx_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R crosses above -50 (recovery) or goes to overbought
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R crosses below -50 (decline) or goes to oversold
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
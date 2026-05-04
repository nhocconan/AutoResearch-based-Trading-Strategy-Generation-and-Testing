#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with 1w ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Enters long when Bull Power > 0 and Bear Power rising from negative
# with volume spike and 1w ADX > 25 (trending market). Enters short when Bear Power < 0 and Bull Power falling from positive
# with volume spike and 1w ADX > 25. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via bull power strength and in bear markets via bear power strength.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_ElderRay_Index_1wADX25_Regime_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (EMA13) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d data for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d data
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Align Elder Ray components to 12h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 1w data for ADX regime filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on 1w data
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = WilderSmoothing(tr, 14)
    dm_plus_smoothed = WilderSmoothing(dm_plus, 14)
    dm_minus_smoothed = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / atr_1w
    di_minus = 100 * dm_minus_smoothed / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    def WilderSmoothing_ADX(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    adx_1w = WilderSmoothing_ADX(dx, 14)
    
    # Align ADX to 12h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power rising from negative AND ADX > 25 AND volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] > bear_power_aligned[i-1] and  # Bear Power rising
                adx_1w_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power falling from positive AND ADX > 25 AND volume spike
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] < bull_power_aligned[i-1] and  # Bull Power falling
                  adx_1w_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power < 0 OR ADX < 20 (regime change)
            if bull_power_aligned[i] < 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power > 0 OR ADX < 20 (regime change)
            if bear_power_aligned[i] > 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
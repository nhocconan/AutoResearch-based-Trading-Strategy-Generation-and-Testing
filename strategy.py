#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d ADX > 25 (trending market)
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d ADX > 25 (trending market)
# Uses 6h timeframe for entries, 1d for regime filtering. Works in bull markets via long signals,
# and in bear markets via short signals. Target: 12-37 trades/year to minimize fee drag.

name = "6h_ElderRay_1dADX25_Regime"
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
    
    # Get 1d data for ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period]) / period
        # Subsequent values
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Long conditions: Bull Power > 0 AND Bear Power rising (less negative than previous)
                if (bull_power[i] > 0 and 
                    i > 50 and bear_power[i] > bear_power[i-1]):
                    signals[i] = 0.25
                    position = 1
                # Short conditions: Bear Power < 0 AND Bull Power falling (less positive than previous)
                elif (bear_power[i] < 0 and 
                      i > 50 and bull_power[i] < bull_power[i-1]):
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: Bull Power <= 0 OR Bear Power stops rising
                if bull_power[i] <= 0 or (i > 50 and bear_power[i] <= bear_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bear Power >= 0 OR Bull Power stops falling
                if bear_power[i] >= 0 or (i > 50 and bull_power[i] >= bull_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging market (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals
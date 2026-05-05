#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1w ADX regime filter and volume confirmation
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND 1w ADX > 25 (strong trend) AND volume > 1.5x 20 EMA
# Short when Bear Power > 0 AND 1w ADX > 25 (strong trend) AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.
# Uses 1w for HTF regime to avoid weak/choppy markets and 6h for entry timing.

name = "6h_ElderRay_Power_1wADX_VolumeConfirm"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get weekly OHLC arrays
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA13 for Elder Ray Power
    if len(df_1w) < 13:
        return np.zeros(n)
    
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power components
    bull_power_1w = high_1w - ema_13_1w  # High - EMA13
    bear_power_1w = ema_13_1w - low_1w   # EMA13 - Low
    
    # Align weekly Elder Ray Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    
    # Get 1w data for ADX regime filter
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1w[1:] - high_1w[:-1]])
    down_move = np.concatenate([[np.nan], low_1w[:-1] - low_1w[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            first_avg = np.nansum(data[1:period+1]) / period
            result[period] = first_avg
            # Subsequent values: Wilder's smoothing
            for i in range(period+1, len(data)):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
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
            # Long conditions: Bull Power > 0 AND 1w ADX > 25 (strong trend) AND volume spike
            if (bull_power_aligned[i] > 0 and 
                adx_1w_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND 1w ADX > 25 (strong trend) AND volume spike
            elif (bear_power_aligned[i] > 0 and 
                  adx_1w_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR 1w ADX <= 20 (weakening trend)
            if (bull_power_aligned[i] <= 0 or 
                adx_1w_aligned[i] <= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR 1w ADX <= 20 (weakening trend)
            if (bear_power_aligned[i] <= 0 or 
                adx_1w_aligned[i] <= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
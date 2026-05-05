#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX25 Trend Filter + Volume Spike
# Long when Bull Power > 0 AND Bear Power < 0 AND ADX(12h) > 25 AND volume > 2.0x 20-period average
# Short when Bull Power < 0 AND Bear Power > 0 AND ADX(12h) > 25 AND volume > 2.0x 20-period average
# Exit when Bull Power and Bear Power have same sign (both >0 or both <0) OR ADX(12h) < 20
# Elder Ray measures bull/bear power via EMA13, effective in ranging markets with trend filter
# 12h ADX25 ensures we only trade when higher timeframe trend is strong
# Volume spike confirms institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "6h_ElderRay_12hADX25_Trend_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for ADX25 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    def WilderSmooth(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period and not np.isnan(data[period-1]):
            result[period-1] = np.nansum(data[:period])
        else:
            return result
        # Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmooth(tr, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, (dm_plus_14 / tr14) * 100, 0)
    di_minus = np.where(tr14 != 0, (dm_minus_14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderSmooth(dx, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Elder Ray (Bull Power and Bear Power) using EMA13
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND ADX(12h) > 25 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bull Power < 0 AND Bear Power > 0 AND ADX(12h) > 25 AND volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power and Bear Power same sign (both >0 or both <0) OR ADX(12h) < 20
            if ((bull_power[i] > 0 and bear_power[i] > 0) or 
                (bull_power[i] < 0 and bear_power[i] < 0) or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power and Bear Power same sign (both >0 or both <0) OR ADX(12h) < 20
            if ((bull_power[i] > 0 and bear_power[i] > 0) or 
                (bull_power[i] < 0 and bear_power[i] < 0) or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
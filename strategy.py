#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Keltner Channel breakout with 1-day ADX trend filter and volume confirmation
# Uses ATR-based channel for volatility adaptation, ADX for trend strength, and volume spike for confirmation
# Designed for low-frequency trades (<150 total) to minimize fee drift and work in both bull/bear markets
# Works in bull by catching breakouts, in bear by filtering with ADX and requiring volume confirmation

name = "12h_Keltner_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ADX (trend strength)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Welles Wilder smoothing (alpha = 1/period)
    def Wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = Wilder_smoothing(tr, 14)
    dm_plus_smooth = Wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = Wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = Wilder_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12-hour Keltner Channel (20-period EMA, 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_12h = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_keltner = ema_20 + 2 * atr_12h
    lower_keltner = ema_20 - 2 * atr_12h
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Keltner with ADX > 25 and volume spike
            if (close[i] > upper_keltner[i] and 
                adx_12h[i] > 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Keltner with ADX > 25 and volume spike
            elif (close[i] < lower_keltner[i] and 
                  adx_12h[i] > 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Keltner or ADX weakens
            if (close[i] < lower_keltner[i] or 
                adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Keltner or ADX weakens
            if (close[i] > upper_keltner[i] or 
                adx_12h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
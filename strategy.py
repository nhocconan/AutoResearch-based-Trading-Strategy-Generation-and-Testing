#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    di_plus_1d = wilder_smooth(dm_plus, 14)
    di_minus_1d = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    di_sum = di_plus_1d + di_minus_1d
    dx = np.where(di_sum != 0, 100 * np.abs(di_plus_1d - di_minus_1d) / di_sum, 0)
    adx_1d = wilder_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 55-period Donchian channels (wider for fewer, stronger signals)
    donch_high_1d = pd.Series(high_1d).rolling(window=55, min_periods=55).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=55, min_periods=55).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 20-period average volume
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour  # pre-compute before loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_1d_aligned[i]
        donch_high_val = donch_high_1d_aligned[i]
        donch_low_val = donch_low_1d_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (ADX > 30), price breaks above 55-period Donchian high, volume above average
            if adx_val > 30 and close_val > donch_high_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend (ADX > 30), price breaks below 55-period Donchian low, volume above average
            elif adx_val > 30 and close_val < donch_low_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 55-period Donchian low or trend weakens (ADX < 25)
            if close_val < donch_low_val or adx_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 55-period Donchian high or trend weakens (ADX < 25)
            if close_val > donch_high_val or adx_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_ADX_Donchian55_Volume_Session_v2
# Uses daily ADX for trend strength filter (ADX > 30 for entry, < 25 for exit)
# Uses daily Donchian(55) breakouts for entry (wider channel = fewer, stronger signals)
# Requires volume confirmation above 20-period average
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level or trend weakens
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_ADX_Donchian55_Volume_Session_v2"
timeframe = "4h"
leverage = 1.0
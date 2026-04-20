#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ADX for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
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
    
    atr_12h = wilder_smooth(tr, 14)
    di_plus_12h = wilder_smooth(dm_plus, 14)
    di_minus_12h = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    di_sum = di_plus_12h + di_minus_12h
    dx = np.where(di_sum != 0, 100 * np.abs(di_plus_12h - di_minus_12h) / di_sum, 0)
    adx_12h = wilder_smooth(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 20-period Donchian channels
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 20-period average volume
    volume_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour  # Pre-compute before loop
    
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
        adx_val = adx_12h_aligned[i]
        donch_high_val = donch_high_12h_aligned[i]
        donch_low_val = donch_low_12h_aligned[i]
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
            # Long: ADX > 25 (trending), price breaks above Donchian high, volume above average
            if adx_val > 25 and close_val > donch_high_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending), price breaks below Donchian low, volume above average
            elif adx_val > 25 and close_val < donch_low_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or ADX < 20 (trend weakening)
            if close_val < donch_low_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or ADX < 20 (trend weakening)
            if close_val > donch_high_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_ADX_Donchian_Breakout_Volume_Session_12h
# Uses 12h ADX for trend strength filter (ADX > 25)
# Uses 12h Donchian(20) breakouts for entry
# Requires volume confirmation above 20-period average
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level or trend weakens (ADX < 20)
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_ADX_Donchian_Breakout_Volume_Session_12h"
timeframe = "4h"
leverage = 1.0
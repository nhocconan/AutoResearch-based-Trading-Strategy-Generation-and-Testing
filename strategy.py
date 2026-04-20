#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 30-period weekly ADX for trend strength
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    
    atr_1w = wilder_smooth(tr, 30)
    di_plus_1w = wilder_smooth(dm_plus, 30)
    di_minus_1w = wilder_smooth(dm_minus, 30)
    
    di_sum = di_plus_1w + di_minus_1w
    dx = np.where(di_sum != 0, 100 * np.abs(di_plus_1w - di_minus_1w) / di_sum, 0)
    adx_1w = wilder_smooth(dx, 30)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily Donchian channels (55-period for longer-term breaks)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donch_high_55 = pd.Series(high_1d).rolling(window=55, min_periods=55).max().values
    donch_low_55 = pd.Series(low_1d).rolling(window=55, min_periods=55).min().values
    donch_high_55_aligned = align_htf_to_ltf(prices, df_1d, donch_high_55)
    donch_low_55_aligned = align_htf_to_ltf(prices, df_1d, donch_low_55)
    
    # Calculate 50-period average volume
    volume_1d = df_1d['volume'].values
    vol_avg_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    vol_avg_50_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_50)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_1w_aligned[i]
        donch_high_val = donch_high_55_aligned[i]
        donch_low_val = donch_low_55_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_50_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong weekly trend (ADX > 25), price breaks above 55-day high, volume above average
            if adx_val > 25 and close_val > donch_high_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: Strong weekly trend (ADX > 25), price breaks below 55-day low, volume above average
            elif adx_val > 25 and close_val < donch_low_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 55-day low or trend weakens (ADX < 20)
            if close_val < donch_low_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 55-day high or trend weakens (ADX < 20)
            if close_val > donch_high_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyADX_Donchian55_Breakout_Volume_Session
# Uses weekly ADX (30-period) for trend strength filter (ADX > 25)
# Uses daily Donchian(55) breakouts for entry
# Requires volume confirmation above 50-period average
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level or trend weakens (ADX < 20)
# Designed for 1d timeframe with ~8-18 trades/year
name = "1d_WeeklyADX_Donchian55_Breakout_Volume_Session"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Supertrend (ATR=10, multiplier=3) with 1d ADX filter (>25) and volume confirmation.
Long when Supertrend turns green, ADX>25, and volume > 1.5x 20-period average.
Short when Supertrend turns red, ADX>25, and volume > 1.5x 20-period average.
Exit when Supertrend reverses.
Uses ADX to avoid whipsaws in ranging markets and Supertrend for clear trend signals.
Designed for low trade frequency (15-30/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ADX on daily data
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_smooth = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate Supertrend on 6h data
    atr_period = 10
    multiplier = 3
    
    # True Range for ATR
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    
    # ATR using Wilder's smoothing
    atr = wilders_smooth(tr_6h, atr_period)
    
    # Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Supertrend calculation
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    if not np.isnan(atr[atr_period-1]):
        upper_band[atr_period-1] = hl2[atr_period-1] + (multiplier * atr[atr_period-1])
        lower_band[atr_period-1] = hl2[atr_period-1] - (multiplier * atr[atr_period-1])
        supertrend[atr_period-1] = upper_band[atr_period-1]
        direction[atr_period-1] = -1  # Start in downtrend
    
    for i in range(atr_period, n):
        if np.isnan(atr[i]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        # Update bands
        upper_band[i] = upper_band[i-1]
        lower_band[i] = lower_band[i-1]
        
        if close[i-1] > upper_band[i-1]:
            upper_band[i] = hl2[i] + (multiplier * atr[i])
        if close[i-1] < lower_band[i-1]:
            lower_band[i] = hl2[i] - (multiplier * atr[i])
            
        # Determine trend
        if close[i] > upper_band[i]:
            direction[i] = 1  # Uptrend
        elif close[i] < lower_band[i]:
            direction[i] = -1  # Downtrend
        else:
            direction[i] = direction[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
                
        supertrend[i] = upper_band[i] if direction[i] == -1 else lower_band[i]
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(atr_period, 14) + 14, n):  # Start after all lookbacks
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend, ADX>25, volume spike
            if (direction[i] == 1 and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, ADX>25, volume spike
            elif (direction[i] == -1 and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Supertrend reverses
            if (position == 1 and direction[i] == -1) or \
               (position == -1 and direction[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Supertrend_ADX25_Volume"
timeframe = "6h"
leverage = 1.0
#%%
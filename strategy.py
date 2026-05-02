#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation (>1.5x average), and 1d ADX regime filter (ADX > 25)
# Donchian breakouts capture momentum in trending markets. EMA50 on 12h ensures alignment with intermediate trend.
# Volume confirmation avoids false breakouts. ADX > 25 ensures we only trade in trending regimes, avoiding whipsaws in ranges.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# Primary timeframe: 6h, HTF: 12h for EMA50 and Donchian reference, 1d for ADX regime.

name = "6h_Donchian20_Breakout_12hEMA50_Volume_ADX"
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
    
    # Calculate Donchian channels (20-period) from 6h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ADX for regime filter (ADX > 25 = trending)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    atr[atr_period-1] = np.mean(tr[:atr_period])
    dm_plus_smooth[atr_period-1] = np.mean(dm_plus[:atr_period])
    dm_minus_smooth[atr_period-1] = np.mean(dm_minus[:atr_period])
    
    # Wilder's smoothing
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (atr_period - 1) + dm_plus[i]) / atr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (atr_period - 1) + dm_minus[i]) / atr_period
    
    # Avoid division by zero
    dmi_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    dmi_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((dmi_plus + dmi_minus) != 0, 100 * np.abs(dmi_plus - dmi_minus) / (dmi_plus + dmi_minus), 0)
    adx = np.zeros_like(dx)
    
    # Smooth DX to get ADX
    adx[2*atr_period-1] = np.mean(dx[atr_period:2*atr_period])
    for i in range(2*atr_period, len(dx)):
        adx[i] = (adx[i-1] * (atr_period - 1) + dx[i]) / atr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND price > 12h EMA50 AND volume spike AND ADX > 25
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price < 12h EMA50 AND volume spike AND ADX > 25
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian low OR price < 12h EMA50
            if close[i] < lowest_low[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian high OR price > 12h EMA50
            if close[i] > highest_high[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
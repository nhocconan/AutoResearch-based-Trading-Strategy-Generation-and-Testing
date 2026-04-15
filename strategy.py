#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX trend filter
# Long when price breaks above 4h Donchian upper band + volume spike + ADX > 25
# Short when price breaks below 4h Donchian lower band + volume spike + ADX > 25
# Exit when price crosses 40-period EMA or ADX < 20 (ranging market)
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian bands (20-period)
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # 40-period EMA for exit
    ema_len = 40
    ema = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # ADX (14-period) on 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(close_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(close_1d, 1)), 
                        np.maximum(np.roll(close_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x median of last 20 bars
        vol_ma = np.median(volume[max(0, i-20):i+1])
        vol_ok = volume[i] > 1.5 * vol_ma
        
        # Long entry: price breaks above Donchian upper + volume + ADX > 25
        if (close[i] > upper[i] and vol_ok and adx_aligned[i] > 25 and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower + volume + ADX > 25
        elif (close[i] < lower[i] and vol_ok and adx_aligned[i] > 25 and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price crosses 40-period EMA or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < ema[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0
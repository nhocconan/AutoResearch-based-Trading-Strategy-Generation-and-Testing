#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX trend filter with 20-period Donchian breakout and volume confirmation
# Long when price breaks above Donchian(20) high AND ADX > 25 AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND ADX > 25 AND volume > 1.5x average
# Exit when price crosses 20-period EMA in opposite direction
# ADX filters for trending markets, Donchian captures breakouts, volume confirms institutional interest.
# Designed to work in both bull and bear markets by capturing strong directional moves.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d (14-period)
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
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate Donchian channels on 1d (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    # Calculate 20-period EMA on 1d for exit
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    
    # Align all 1d indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low.values)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20.values)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20 for Donchian/EMA + buffer)
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema_val = ema_20_aligned[i]
        vol_avg_val = vol_avg_aligned[i]
        close_val = close[i]
        vol = volume[i]
        vol_threshold = vol_avg_val * 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND ADX > 25 AND volume confirmation
            if (close_val > donch_high_val and adx_val > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low AND ADX > 25 AND volume confirmation
            elif (close_val < donch_low_val and adx_val > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 20-period EMA
            if close_val < ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 20-period EMA
            if close_val > ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_ADX_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
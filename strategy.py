#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses Donchian(20) midpoint OR 1d ADX drops below 20 (range)
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-35 trades/year per symbol.
# Donchian provides objective breakout levels, volume confirms institutional participation,
# 1d ADX filters for trending markets to avoid whipsaws in ranging conditions.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeConfirm_1dADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian Channel on 4h data (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to prices timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Trend conditions: ADX > 25 for trending, ADX < 20 for ranging
    trending_1d = adx_1d > 25.0
    ranging_1d = adx_1d < 20.0
    
    # Align 1d trend to 4h timeframe
    trending_1d_aligned = align_htf_to_ltf(prices, df_1d, trending_1d.astype(float))
    ranging_1d_aligned = align_htf_to_ltf(prices, df_1d, ranging_1d.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(trending_1d_aligned[i]) or 
            np.isnan(ranging_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume confirmation AND 1d ADX trending
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                trending_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume confirmation AND 1d ADX trending
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  trending_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR 1d ADX drops to ranging
            if (close[i] < donchian_mid_aligned[i] or 
                ranging_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR 1d ADX drops to ranging
            if (close[i] > donchian_mid_aligned[i] or 
                ranging_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
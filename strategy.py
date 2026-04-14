#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ADX trend filter with 1-day Donchian breakout and volume confirmation
# Long when price breaks above 1-day Donchian(20) high AND ADX(14) > 25 AND volume > 1.5x 20-period average
# Short when price breaks below 1-day Donchian(20) low AND ADX(14) > 25 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the 1-day Donchian channel (opposite band)
# Uses higher timeframe structure (1-day) to reduce whipsaw in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX(14) on 1-day data for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_14_values = adx_14.values
    
    # Align 1-day indicators to 4-hour timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_values)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + 14 for ADX + buffer)
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above 1-day Donchian high + ADX > 25 + volume confirmation
            if (price > high_20_1d_aligned[i] and adx_14_aligned[i] > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below 1-day Donchian low + ADX > 25 + volume confirmation
            elif (price < low_20_1d_aligned[i] and adx_14_aligned[i] > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below 1-day Donchian low (opposite band)
            if price < low_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above 1-day Donchian high (opposite band)
            if price > high_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_ADX_Donchian1D_Volume"
timeframe = "4h"
leverage = 1.0
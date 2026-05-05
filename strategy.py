#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when price breaks below 4h Donchian lower (20) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when price crosses 4h Donchian midpoint OR 1d ADX < 20 (trend weakening)
# Uses Donchian structure for breakouts, ADX for regime filter to avoid whipsaws, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
# Timeframe: 4h (primary)

name = "4h_Donchian20_1dADX_Trend_VolumeSpike_2.0x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period) using previous bar's values to avoid look-ahead
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    # We use rolling window on 4h data, then align
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d_series.diff()
    down_move = low_1d_series.shift(1) - low_1d_series  # positive when low decreases
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and ATR
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d.values
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d.values
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 4h (threshold: 2.0x for balance)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND ADX > 25 (strong trend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 (strong trend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR ADX < 20 (trend weakening)
            if close[i] < donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR ADX < 20 (trend weakening)
            if close[i] > donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
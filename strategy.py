#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX filter and volume confirmation
# Long when price breaks above Donchian upper (20) AND ADX(14) > 25 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20) AND ADX(14) > 25 AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (average of upper/lower)
# Uses ADX to filter for trending markets only, reducing whipsaws in ranging conditions
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_donchian20_12h_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 12h ADX(14) trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = pd.Series(high_12h).rolling(2).max() - pd.Series(low_12h).rolling(2).min()
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = -pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr_12h
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with ADX filter and volume confirmation
            # Long: price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                adx_12h_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  adx_12h_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
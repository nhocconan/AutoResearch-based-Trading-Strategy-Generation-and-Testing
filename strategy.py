#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + 12h ADX trend filter
# Long when price breaks above 4h Donchian upper (20) AND 12h volume > 1.5x average AND 12h ADX > 25
# Short when price breaks below 4h Donchian lower (20) AND 12h volume > 1.5x average AND 12h ADX > 25
# Exit when price returns to 4h Donchian middle (10) or ADX < 20
# Uses 4h timeframe for signal, 12h for volume and trend filters to reduce noise
# Works in both bull/bear markets by trading breakouts with trend and volume confirmation
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_12h_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 12h data for volume and ADX filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean()
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h.values)
    
    # 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean()
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False).mean()
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to Donchian middle OR ADX < 20 (trend weakening)
        if position == 1:  # long position
            if close[i] <= donchian_middle[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_middle[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume and trend confirmation
            # Bullish breakout: price above Donchian upper + volume spike + ADX > 25
            if (close[i] > donchian_upper[i] and 
                volume[i] > 1.5 * vol_ma_12h_aligned[i] and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian lower + volume spike + ADX > 25
            elif (close[i] < donchian_lower[i] and 
                  volume[i] > 1.5 * vol_ma_12h_aligned[i] and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
    
    return signals
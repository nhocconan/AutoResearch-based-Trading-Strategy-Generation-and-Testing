#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian upper channel with volume > 1.5x average and ADX > 25
# Short when price breaks below 12h Donchian lower channel with volume > 1.5x average and ADX > 25
# Exit when price returns to opposite channel or ADX < 20
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12-hour Donchian for structure and 1-day volume/ADX for confirmation
# Target: 70-120 total trades over 4 years (17-30/year)

name = "12h_donchian_1d_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels: 20-period high/low
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 1-day volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ADX(14) for trend strength on 1-day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to lower Donchian or ADX weakens (< 20)
            elif close[i] <= donchian_lower_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to upper Donchian or ADX weakens (< 20)
            elif close[i] >= donchian_upper_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks Donchian with volume spike and strong trend
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma_1d_aligned[i]
            # Strong trend: ADX > 25
            strong_trend = adx_1d_aligned[i] > 25
            
            # Long: price breaks above upper Donchian, volume spike, strong trend
            if (close[i] > donchian_upper_aligned[i] and 
                volume_spike and strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower Donchian, volume spike, strong trend
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_spike and strong_trend):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
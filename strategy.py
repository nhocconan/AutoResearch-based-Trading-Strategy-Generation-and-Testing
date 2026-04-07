#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian high (20) with 1d volume > 1.5x average and ADX > 25
# Short when price breaks below 12h Donchian low (20) with 1d volume > 1.5x average and ADX > 25
# Exit when price returns to 12h Donchian midpoint or ADX < 20
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12-hour price channel and 1-day volume/ADX for confirmation
# Target: 50-120 total trades over 4 years (12-30/year)

name = "12h_donchian20_1d_vol_adx_v1"
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
    
    # 12-hour data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period rolling max/min for Donchian
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned, just shift for completed bar)
    donchian_high = np.roll(high_20, 1)  # Previous completed bar
    donchian_low = np.roll(low_20, 1)    # Previous completed bar
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # 1-day data for volume and ADX confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day volume average (20-period)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) on 1-day for trend strength
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
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(14) for stoploss (using 12h data)
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close, 1))
    tr_12h2[0] = tr_12h1[0]
    tr_12h3[0] = tr_12h1[0]
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Donchian midpoint or ADX weakens (< 20)
            elif close[i] <= (donchian_high[i] + donchian_low[i]) / 2 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Donchian midpoint or ADX weakens (< 20)
            elif close[i] >= (donchian_high[i] + donchian_low[i]) / 2 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks Donchian with volume spike and strong trend
            # Volume spike: > 1.5x average volume
            volume_spike = vol_1d[i] > 1.5 * vol_ma_1d_aligned[i]
            # Strong trend: ADX > 25
            strong_trend = adx_1d_aligned[i] > 25
            
            # Long: price breaks above Donchian high, volume spike, strong trend
            if close[i] > donchian_high[i] and volume_spike and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, volume spike, strong trend
            elif close[i] < donchian_low[i] and volume_spike and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
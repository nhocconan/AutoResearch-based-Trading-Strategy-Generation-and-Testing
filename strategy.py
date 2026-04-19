#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX for trend strength filter (avoid ranging markets)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    # DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d ATR for volatility filter
    atr_1d = tr_smooth  # already smoothed TR
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h Donchian channels (20-period)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        vol = volume[i]
        
        # Trend filter: only trade when 1d ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma[i]
        
        # Entry conditions: Donchian breakout with trend and volume confirmation
        if position == 0:
            # Long: price breaks above 20-period high + trend up + volume
            if price > donch_high_20[i] and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + trend down + volume
            elif price < donch_low_20[i] and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below 20-period low or ATR-based trailing stop
            if price < donch_low_20[i] or price < high[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above 20-period high or ATR-based trailing stop
            if price > donch_high_20[i] or price > low[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
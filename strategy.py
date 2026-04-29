#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# Donchian breakouts capture momentum; volume >1.5x confirms participation; ADX>25 ensures trending market
# Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via ADX trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_VolumeSpike_1dADX25_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # True Range
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'],
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)),
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = 0
    tr_1d = pd.Series(tr_1d)
    
    # Directional Movement
    up_move = df_1d['high'] - np.roll(df_1d['high'], 1)
    down_move = np.roll(df_1d['low'], 1) - df_1d['low']
    up_move[0] = 0
    down_move[0] = 0
    up_move = pd.Series(up_move)
    down_move = pd.Series(down_move)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # Smoothed values
    tr_14 = tr_1d.rolling(window=14, min_periods=14).sum()
    plus_dm_14 = plus_dm.rolling(window=14, min_periods=14).sum()
    minus_dm_14 = minus_dm.rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di_14 = 100 * (plus_dm_14 / tr_14)
    minus_di_14 = 100 * (minus_dm_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14, 14) + 28  # warmup for Donchian, volume, ATR, ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter (ADX > 25)
            if curr_volume_confirm and curr_adx > 25:
                # Bullish entry: price breaks above Donchian upper
                if curr_high > highest_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower
                elif curr_low < lowest_20[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower
            if curr_low < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper
            if curr_high > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
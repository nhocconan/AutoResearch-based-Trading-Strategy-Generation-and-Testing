#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter
# Long when price breaks above Donchian upper band AND volume > 2.5x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below Donchian lower band AND volume > 2.5x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses back to Donchian middle band OR 1d ADX < 20 (range)
# Uses discrete sizing (0.30) to limit fee drag. Target: 15-35 trades/year per symbol.
# Donchian provides objective structure, volume confirms institutional participation, ADX filters for trending markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends, avoids whipsaws in ranging markets.

name = "12h_Donchian20_VolumeSpike_1dADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Trend filters
    uptrend_1d = adx > 25
    ranging_1d = adx < 20
    
    # Align Donchian and ADX to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    ranging_1d_aligned = align_htf_to_ltf(prices, df_1d, ranging_1d.astype(float))
    
    # Volume confirmation: volume > 2.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(ranging_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND volume spike AND 1d uptrend
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below lower band AND volume spike AND 1d uptrend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  uptrend_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to middle band OR 1d ranging
            if (close[i] < donchian_mid_aligned[i] or 
                ranging_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to middle band OR 1d ranging
            if (close[i] > donchian_mid_aligned[i] or 
                ranging_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
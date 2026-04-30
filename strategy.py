#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX(14) regime filter + ATR(14) trailing stop
# Donchian channels provide clear breakout levels; volume confirms breakout strength; 
# 1d ADX > 25 filters for trending markets (works in both bull/bear via breakout direction);
# ATR-based trailing stop limits drawdown. Discrete sizing 0.25 minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_VolumeSpike_ADXregime_ATRStop_v1"
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
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # Smoothed DM and TR
    tr_ma = atr_1d  # already smoothed TR
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_ma)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr_ma)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False).mean()
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    
    # ATR(14) for trailing stop (calculated on 4h data)
    tr_4h1 = pd.Series(high).diff().abs()
    tr_4h2 = (pd.Series(high) - pd.Series(close.shift(1))).abs()
    tr_4h3 = (pd.Series(low) - pd.Series(close.shift(1))).abs()
    tr_4h = pd.concat([tr_4h1, tr_4h2, tr_4h3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14)  # warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_volume_spike = volume_spike[i]
        curr_adx = adx_1d_aligned[i]
        curr_atr = atr_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending regime (ADX > 25)
            if curr_volume_spike and curr_adx > 25:
                # Bullish breakout: price breaks above Donchian high
                if curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = curr_high
                # Bearish breakout: price breaks below Donchian low
                elif curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops from high by 3.0 * ATR
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises from low by 3.0 * ATR
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
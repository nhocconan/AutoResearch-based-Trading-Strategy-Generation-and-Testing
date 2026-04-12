#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Donchian breakouts capture strong trends; 1d volume confirms institutional participation
    # Chop filter avoids whipsaws in ranging markets; discrete sizing reduces fee drag
    # Target: 12-30 trades/year per symbol for 12h timeframe to minimize costs
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for chop filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    tr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14_1d = 100 * pd.Series(plus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14_1d
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
    adx_14_1d = pd.Series(dx_14_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate Donchian channels (20-period) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    highest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_ratio = np.full(n, np.nan)
    for i in range(n):
        if vol_ma_20_1d_aligned[i] > 0:
            vol_ratio[i] = volume_1d_aligned[i] / vol_ma_20_1d_aligned[i]
        else:
            vol_ratio[i] = 1.0
    
    # Get 1d volume aligned for current bar
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_20_12h_aligned[i]) or np.isnan(lowest_20_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Chop regime: ADX < 25 indicates ranging market
        chop_regime = adx_14_1d_aligned[i] < 25
        
        # Breakout conditions with volume confirmation
        breakout_long = (close[i] > highest_20_12h_aligned[i]) and \
                        (vol_ratio[i] > 1.5) and \
                        uptrend and \
                        (not chop_regime)  # Only trade breakouts in trending markets
        
        breakout_short = (close[i] < lowest_20_12h_aligned[i]) and \
                         (vol_ratio[i] > 1.5) and \
                         downtrend and \
                         (not chop_regime)  # Only trade breakouts in trending markets
        
        # Exit conditions: return to midpoint of Donchian channel
        midpoint_20 = (highest_20_12h_aligned[i] + lowest_20_12h_aligned[i]) / 2.0
        long_exit = close[i] < midpoint_20
        short_exit = close[i] > midpoint_20
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_vol_adx_v1"
timeframe = "12h"
leverage = 1.0
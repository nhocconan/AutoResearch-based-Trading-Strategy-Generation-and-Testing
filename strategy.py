#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
    # Long when: price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND volume > 1.5x avg volume
    # Short when: price breaks below Donchian(20) low AND 1d ADX > 25 (trending) AND volume > 1.5x avg volume
    # Exit when: price crosses Donchian midpoint OR ADX drops below 20 (range) OR volume < avg
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via ADX trend filter ensuring we only trade strong trends.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original indices
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    # Initialize first values
    if not np.isnan(tr[period]):
        atr[period] = np.nanmean(tr[1:period+1])
        plus_di[period] = 100 * np.nanmean(plus_dm[1:period+1]) / atr[period] if atr[period] != 0 else 0
        minus_di[period] = 100 * np.nanmean(minus_dm[1:period+1]) / atr[period] if atr[period] != 0 else 0
    
    # Wilder smoothing
    for i in range(period + 1, len(tr)):
        if np.isnan(atr[i-1]):
            atr[i] = np.nan
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        if np.isnan(plus_di[i-1]) or np.isnan(minus_di[i-1]):
            plus_di[i] = np.nan
            minus_di[i] = np.nan
            dx[i] = np.nan
        else:
            plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
        
        # ADX: smoothed DX
        if np.isnan(dx[i]) or np.isnan(adx[i-1]):
            adx[i] = np.nan
        elif np.isnan(adx[i-1]):
            adx[i] = np.nanmean(dx[period+1:i+1]) if i >= period+1 else np.nan
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) channels on 6h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions
        long_entry = long_breakout and strong_trend and vol_ok and position != 1
        short_entry = short_breakout and strong_trend and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR ADX drops below 20 (range) OR volume < avg
        exit_long = (close[i] < donchian_mid[i]) or (adx_aligned[i] < 20) or (volume[i] < vol_ma[i])
        exit_short = (close[i] > donchian_mid[i]) or (adx_aligned[i] < 20) or (volume[i] < vol_ma[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0
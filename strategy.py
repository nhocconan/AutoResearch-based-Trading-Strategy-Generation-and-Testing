#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX trend filter
    # Long: price breaks above Donchian upper + volume > 1.5x 20-period average + ADX > 25
    # Short: price breaks below Donchian lower + volume > 1.5x 20-period average + ADX > 25
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation and ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume average (20-period) for confirmation
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX for trend filter (ADX > 25 = trending)
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # prepend NaN for index alignment
    
    # +DM and -DM
    up_move = np.concatenate([[np.nan], high_12h[1:] - high_12h[:-1]])
    down_move = np.concatenate([[np.nan], low_12h[:-1] - low_12h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    tr_period = np.zeros_like(tr)
    plus_dm_period = np.zeros_like(plus_dm)
    minus_dm_period = np.zeros_like(minus_dm)
    
    # Initial values (simple average)
    tr_period[period] = np.nansum(tr[1:period+1])
    plus_dm_period[period] = np.nansum(plus_dm[1:period+1])
    minus_dm_period[period] = np.nansum(minus_dm[1:period+1])
    
    # Wilder's smoothing
    for i in range(period+1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
        minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
    
    # +DI and -DI
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = np.zeros_like(tr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Replace division by zero with 0
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    # Smoothed ADX
    adx = np.zeros_like(dx)
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 12h indicators to 4h timeframe
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for stoploss
    atr_4h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_4h[i] = tr
        else:
            atr_4h[i] = 0.93 * atr_4h[i-1] + 0.07 * tr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average from 12h
        vol_avg_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_4h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_4h[i]
        
        # Trend filter: 12h ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_upper[i]) and volume_confirmed and trending
        breakout_short = (close[i] < donchian_lower[i]) and volume_confirmed and trending
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_4h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_price_channel_breakout_v1
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian upper channel (20-period) with volume confirmation and price above 200-period EMA (bullish trend filter). Enter short when price breaks below Donchian lower channel with volume confirmation and price below 200-period EMA. Exit when price crosses the 20-period EMA (trend reversal). Uses 12h ADX > 25 to ensure trending market conditions. Designed for 20-40 trades/year to minimize fee dust while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (14-period)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[13] = np.mean(tr[:14])
    dm_plus_smooth[13] = np.mean(dm_plus[:14])
    dm_minus_smooth[13] = np.mean(dm_minus[:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First 14-period average of DX
    
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: 12h ADX > 25 indicates trending market
        trend_ok = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price crosses below 20-period EMA (trend reversal)
            if close[i] < ema_20[i] and close[i-1] >= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period EMA (trend reversal)
            if close[i] > ema_20[i] and close[i-1] <= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok and trend_ok:
                # Long: price breaks above Donchian upper channel with price above 200 EMA
                if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                    # Additional trend filter: price above 200-period EMA
                    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
                    if not np.isnan(ema_200[i]) and close[i] > ema_200[i]:
                        position = 1
                        signals[i] = 0.30
                # Short: price breaks below Donchian lower channel with price below 200 EMA
                elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                    # Additional trend filter: price below 200-period EMA
                    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
                    if not np.isnan(ema_200[i]) and close[i] < ema_200[i]:
                        position = -1
                        signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX(14) regime filter
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# ADX(14) > 25 filters for trending markets only, avoiding range-bound whipsaws
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian provides objective price channels, volume confirms breakout validity
# ADX regime filter ensures trades only occur in favorable trending conditions
# Works in both bull and bear markets by only taking trades in direction of trend

name = "4h_Donchian20_Volume_ADX25_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Calculate ADX(14) for regime filtering
    # ADX calculation requires +DI and -DI
    # +DI = 100 * EWMA of (+DM / TR) over 14 periods
    # -DI = 100 * EWMA of (-DM / TR) over 14 periods
    # ADX = 100 * EWMA of (|+DI - -DI| / (+DI + -DI)) over 14 periods
    
    # Calculate True Range (TR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement (+DM and -DM)
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Calculate smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    tr_m = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_m = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_m = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * dm_plus_m / tr_m
    minus_di = 100 * dm_minus_m / tr_m
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # ADX > 25 indicates trending market
    adx_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(adx_trend[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Donchian upper + volume confirm + ADX > 25 (trending)
            if close[i] > donchian_upper[i] and volume_confirm[i] and adx_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian lower + volume confirm + ADX > 25 (trending)
            elif close[i] < donchian_lower[i] and volume_confirm[i] and adx_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Donchian lower (breakdown) or ADX < 20 (trend weakening)
            if close[i] < donchian_lower[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Donchian upper (breakout) or ADX < 20 (trend weakening)
            if close[i] > donchian_upper[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
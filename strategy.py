#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14022_12h_donchian20_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volume confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_1d_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # 12h data for Donchian and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_1d_ma_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
            continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
            continue
        
        # Volume confirmation: daily volume > 1.5x 20-day average
        volume_ok = volume_1d_ma_aligned[i] > 0 and volume_1d_ma_aligned[i] > (volume_1d_ma_aligned[i] * 0)  # Always true for alignment, use current volume
        volume_ok = volume[i] > (volume_1d_ma_aligned[i] * 1.5) if not np.isnan(volume_1d_ma_aligned[i]) else False
        
        # Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_ok:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_down and volume_ok:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        else:
            # Maintain position if not stopped out
            signals[i] = position * 0.30
    
    return signals
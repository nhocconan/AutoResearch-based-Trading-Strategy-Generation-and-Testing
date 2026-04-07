#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume + Chop Filter
# Hypothesis: Donchian(20) breakouts capture momentum in trending markets.
# Volume confirms institutional participation. Chop filter avoids whipsaws in ranging markets.
# Works in both bull and bear markets by trading breakouts in direction of trend.
# 4h timeframe balances responsiveness and noise. Target: 20-50 trades/year (80-200 over 4 years).
name = "4h_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for chop filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Chop filter: Chop(14) > 61.8 = ranging (avoid), Chop < 38.2 = trending (trade)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI for Chop calculation
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Get Chop values aligned to 4h
    chop_12h = chop
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is trending (Chop < 38.2)
        if chop_12h_aligned[i] >= 38.2:
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above Donchian high with volume
        if close[i] > donchian_high[i-1] and vol_filter[i]:
            signals[i] = 0.25
        # Short breakdown: price breaks below Donchian low with volume
        elif close[i] < donchian_low[i-1] and vol_filter[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals
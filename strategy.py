#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band AND 1d EMA50 uptrend AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND 1d EMA50 downtrend AND volume > 2.0x 20-period average.
Exit when price retouches Donchian midpoint or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to balance return and risk. Targets 12-37 trades/year per symbol.
Donchian channels provide clear breakout levels, 1d EMA50 ensures alignment with daily trend,
volume filters weak breakouts. Designed for 6h timeframe to reduce trade frequency and fee drag
while capturing multi-day swings. Works in both bull (trend continuation) and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian levels from 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 6h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND 1d EMA50 uptrend AND volume spike
            if (price > upper and 
                close[i] > ema50 and  # Current close above EMA50 for uptrend
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian lower AND 1d EMA50 downtrend AND volume spike
            elif (price < lower and 
                  close[i] < ema50 and  # Current close below EMA50 for downtrend
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Donchian midpoint
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0
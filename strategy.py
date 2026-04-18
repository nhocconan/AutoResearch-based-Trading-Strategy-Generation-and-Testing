#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Donchian channel breakouts on 12h timeframe with volume confirmation and daily EMA50 trend filter capture momentum moves in both bull and bear markets. Designed for 12-37 trades/year (50-150 total over 4 years) with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Donchian channel (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Volume spike: 2x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (12h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above EMA50 (uptrend)
            if price > upper and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below EMA50 (downtrend)
            elif price < lower and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or ATR trailing stop
            if price <= lower or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or ATR trailing stop
            if price >= upper or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_EMA50"
timeframe = "12h"
leverage = 1.0